"""
Workspaces: a named set of apps (any category, Play Store and/or App Store)
analysed together with one shared theme taxonomy.

Layout on disk:
    data/workspaces/<slug>/
        workspace.json      # config: apps, taxonomy, country, category
        raw/<app_key>.csv   # scraped reviews per app
        app_meta.json       # store metadata snapshot per app
        cache.json          # LLM classification cache (keyed by review hash)
        classified.csv      # merged + classified reviews
        brief.md            # last AI executive brief
"""
import json
import re
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

WORKSPACES_DIR = Path("data/workspaces")

APP_PALETTE = [
    "#2563eb", "#0891b2", "#059669", "#d97706", "#7c3aed",
    "#db2777", "#dc2626", "#4f46e5", "#0d9488", "#65a30d",
]


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", str(text).strip().lower()).strip("-")
    return s or "workspace"


def app_key(platform: str, app_id: str) -> str:
    return f"{platform}_{re.sub(r'[^a-zA-Z0-9]+', '_', str(app_id))}"


@dataclass
class AppRef:
    name: str            # short display name, e.g. "Swiggy"
    platform: str        # "play" | "appstore"
    app_id: str          # package name or numeric App Store id
    icon: str = ""
    developer: str = ""

    @property
    def key(self) -> str:
        return app_key(self.platform, self.app_id)


@dataclass
class Workspace:
    name: str
    category: str = "General"
    country: str = "in"
    lang: str = "en"
    apps: list = field(default_factory=list)       # list[AppRef]
    taxonomy: list = field(default_factory=list)   # list[dict]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # ── paths ───────────────────────────────────────────
    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def root(self) -> Path:
        return WORKSPACES_DIR / self.slug

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def cache_path(self) -> Path:
        return self.root / "cache.json"

    @property
    def classified_path(self) -> Path:
        return self.root / "classified.csv"

    @property
    def meta_path(self) -> Path:
        return self.root / "app_meta.json"

    @property
    def brief_path(self) -> Path:
        return self.root / "brief.md"

    # ── helpers ─────────────────────────────────────────
    def app_colors(self) -> dict:
        return {a.name: APP_PALETTE[i % len(APP_PALETTE)] for i, a in enumerate(self.apps)}

    def theme_labels(self) -> dict:
        return {t["key"]: t["label"] for t in self.taxonomy}

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        with open(self.root / "workspace.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_meta(self) -> dict:
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def save_meta(self, meta: dict):
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "Workspace":
        apps = [AppRef(**a) for a in d.get("apps", [])]
        return cls(
            name=d["name"],
            category=d.get("category", "General"),
            country=d.get("country", "in"),
            lang=d.get("lang", "en"),
            apps=apps,
            taxonomy=d.get("taxonomy", []),
            created_at=d.get("created_at", ""),
        )


def list_workspaces() -> list[Workspace]:
    ensure_legacy_migrated()
    out = []
    if not WORKSPACES_DIR.exists():
        return out
    for p in sorted(WORKSPACES_DIR.iterdir()):
        cfg = p / "workspace.json"
        if cfg.exists():
            try:
                out.append(Workspace.from_dict(json.loads(cfg.read_text(encoding="utf-8"))))
            except Exception:
                continue
    return out


def load_workspace(slug: str) -> Workspace | None:
    cfg = WORKSPACES_DIR / slug / "workspace.json"
    if not cfg.exists():
        return None
    return Workspace.from_dict(json.loads(cfg.read_text(encoding="utf-8")))


def delete_workspace(slug: str):
    target = WORKSPACES_DIR / slug
    if target.exists() and target.parent == WORKSPACES_DIR:
        shutil.rmtree(target)


# ── Legacy migration ───────────────────────────────────────────────
LEGACY_APPS = [
    ("PhonePe", "com.phonepe.app", "phonepe"),
    ("Paytm", "net.one97.paytm", "paytm"),
    ("Google Pay", "com.google.android.apps.nbu.paisa.user", "google_pay"),
]


def ensure_legacy_migrated():
    """
    One-time: turn the original fintech-only dataset (data/raw, data/classified)
    into the 'Indian UPI Apps' workspace. Legacy files are copied, never moved,
    and the LLM cache is reused so nothing gets re-classified.
    """
    from src.taxonomy import get_preset

    legacy_raw = Path("data/raw")
    if not legacy_raw.exists():
        return
    ws = Workspace(name="Indian UPI Apps", category="Fintech / Payments", country="in", lang="en")
    if (ws.root / "workspace.json").exists():
        return
    if not any(legacy_raw.glob("*_reviews.csv")):
        return

    ws.apps = [AppRef(name=n, platform="play", app_id=pid) for n, pid, _ in LEGACY_APPS]
    ws.taxonomy = get_preset("Fintech / Payments")
    ws.save()

    for name, pid, fname in LEGACY_APPS:
        src = legacy_raw / f"{fname}_reviews.csv"
        if src.exists():
            shutil.copy(src, ws.raw_dir / f"{app_key('play', pid)}.csv")

    # Reuse the legacy LLM cache, but drop entries that were silent parse failures
    # (the old parser stored them as theme='other' / "No summary provided."), so
    # those reviews get re-classified instead of polluting the analysis.
    # The legacy classified CSV is not copied: it was mostly built from those failures.
    legacy_cache = Path("data/classified/cache.json")
    if legacy_cache.exists():
        from src.llm_classify import taxonomy_fingerprint
        bad = {"No summary provided.", "Failed to parse API response", ""}
        cache = json.loads(legacy_cache.read_text(encoding="utf-8"))
        cache = {k: v for k, v in cache.items() if isinstance(v, dict) and v.get("one_line_summary", "") not in bad}
        cache["__taxonomy__"] = taxonomy_fingerprint(ws.taxonomy)
        ws.cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
