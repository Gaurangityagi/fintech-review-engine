import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from pathlib import Path

MANUAL_LABELS_PATH = Path("data/validation/manual_labels.csv")

def create_validation_sample(df: pd.DataFrame, n: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Create a stratified random sample of n reviews for manual labeling.
    Save to data/validation/manual_labels.csv with columns:
    review_id, app_name, review_text, rating, manual_theme, manual_severity
    (manual_theme and manual_severity left blank for human to fill in)
    """
    MANUAL_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Stratified random sample if app_name is available
    if 'app_name' in df.columns:
        try:
            # Attempt to stratify by app_name
            sample = df.groupby('app_name', group_keys=False).apply(
                lambda x: x.sample(min(len(x), n // df['app_name'].nunique()), random_state=seed)
            )
            if len(sample) < n:
                remaining = df.drop(sample.index).sample(n - len(sample), random_state=seed)
                sample = pd.concat([sample, remaining])
        except ValueError:
            sample = df.sample(n, random_state=seed)
    else:
        sample = df.sample(n, random_state=seed)
        
    out_df = pd.DataFrame()
    out_df['review_id'] = sample.get('review_id', [f"rev_{i}" for i in range(len(sample))])
    out_df['app_name'] = sample.get('app_name', 'Unknown')
    out_df['review_text'] = sample.get('review_text', '')
    out_df['rating'] = sample.get('rating', 0)
    out_df['manual_theme'] = ''
    out_df['manual_severity'] = ''
    
    out_df.to_csv(MANUAL_LABELS_PATH, index=False)
    return out_df

def load_manual_labels() -> pd.DataFrame:
    """Load the manually-labeled validation sample."""
    if not MANUAL_LABELS_PATH.exists():
        raise FileNotFoundError(f"Validation file {MANUAL_LABELS_PATH} not found.")
    return pd.read_csv(MANUAL_LABELS_PATH)

def compute_agreement(llm_df: pd.DataFrame, manual_df: pd.DataFrame) -> dict:
    """
    Compute agreement metrics between LLM and manual labels:
    - theme_accuracy: simple accuracy for theme classification
    - severity_accuracy: simple accuracy for severity classification  
    - theme_kappa: Cohen's kappa for theme
    - severity_kappa: Cohen's kappa for severity
    - confusion_matrix: for themes
    - disagreements: list of reviews where LLM and manual labels differ,
      with both labels shown
    Return dict with all metrics.
    """
    if 'review_id' in llm_df.columns and 'review_id' in manual_df.columns:
        merged = manual_df.merge(llm_df, on='review_id', suffixes=('_manual', '_llm'))
    else:
        merged = manual_df.merge(llm_df, on='review_text', suffixes=('_manual', '_llm'))
        
    if len(merged) == 0:
        return {}
        
    # Get labels
    manual_themes = merged.get('manual_theme', merged.get('manual_theme_manual')).fillna('other')
    llm_themes = merged.get('theme', merged.get('theme_llm')).fillna('other')
    
    manual_severities = merged.get('manual_severity', merged.get('manual_severity_manual')).fillna('medium')
    llm_severities = merged.get('severity', merged.get('severity_llm')).fillna('medium')
    
    # Accuracies
    theme_accuracy = accuracy_score(manual_themes, llm_themes)
    severity_accuracy = accuracy_score(manual_severities, llm_severities)
    
    # Kappas
    theme_kappa = cohen_kappa_score(manual_themes, llm_themes)
    severity_kappa = cohen_kappa_score(manual_severities, llm_severities)
    
    # Confusion Matrix
    labels = sorted(list(set(manual_themes).union(set(llm_themes))))
    cm = confusion_matrix(manual_themes, llm_themes, labels=labels)
    cm_dict = {
        "labels": labels,
        "matrix": cm.tolist()
    }
    
    # Disagreements
    disagreements = []
    for idx, row in merged.iterrows():
        m_theme = manual_themes.iloc[idx]
        l_theme = llm_themes.iloc[idx]
        
        m_sev = manual_severities.iloc[idx]
        l_sev = llm_severities.iloc[idx]
        
        if m_theme != l_theme or m_sev != l_sev:
            disagreements.append({
                "review_text": row.get('review_text', ''),
                "manual_theme": m_theme,
                "llm_theme": l_theme,
                "manual_severity": m_sev,
                "llm_severity": l_sev
            })
            
    return {
        "theme_accuracy": theme_accuracy,
        "severity_accuracy": severity_accuracy,
        "theme_kappa": theme_kappa,
        "severity_kappa": severity_kappa,
        "confusion_matrix": cm_dict,
        "disagreements": disagreements
    }

def generate_validation_report(agreement: dict) -> str:
    """
    Generate a markdown report string from agreement metrics.
    Include:
    - Overall accuracy and kappa scores
    - Top disagreement patterns
    - 2-3 specific example disagreements with analysis
    """
    if not agreement:
        return "No validation data to report on."
        
    report = []
    report.append("# Classification Validation Report")
    report.append("")
    report.append("## Overall Metrics")
    report.append(f"- **Theme Accuracy:** {agreement.get('theme_accuracy', 0):.2%}")
    report.append(f"- **Theme Cohen's Kappa:** {agreement.get('theme_kappa', 0):.2f}")
    report.append(f"- **Severity Accuracy:** {agreement.get('severity_accuracy', 0):.2%}")
    report.append(f"- **Severity Cohen's Kappa:** {agreement.get('severity_kappa', 0):.2f}")
    report.append("")
    
    disagreements = agreement.get("disagreements", [])
    report.append(f"## Disagreements ({len(disagreements)} total found)")
    report.append("Here are a few example disagreements:")
    
    for d in disagreements[:3]:
        report.append(f"\n**Review:** {d['review_text']}")
        if d['manual_theme'] != d['llm_theme']:
            report.append(f"- Theme Mismatch -> Manual: `{d['manual_theme']}` | LLM: `{d['llm_theme']}`")
        if d['manual_severity'] != d['llm_severity']:
            report.append(f"- Severity Mismatch -> Manual: `{d['manual_severity']}` | LLM: `{d['llm_severity']}`")
            
    return "\n".join(report)


# ═══════════════════════════════════════════════════════════════
#  WORKSPACE-GENERIC VALIDATION (human-in-the-loop, any taxonomy)
# ═══════════════════════════════════════════════════════════════
def workspace_label_sample(classified: pd.DataFrame, n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Stratified (by app) sample for manual labelling; LLM labels are hidden to avoid anchoring bias."""
    d = classified.dropna(subset=["theme"])
    d = d[d["review_text"].astype(str).str.len() > 15]
    per_app = max(1, n // max(1, d["app_name"].nunique()))
    sample = d.groupby("app_name", group_keys=False).apply(lambda g: g.sample(min(len(g), per_app), random_state=seed))
    out = sample[["review_id", "app_name", "rating", "review_text"]].copy()
    out["manual_theme"] = None
    out["manual_severity"] = None
    return out.reset_index(drop=True)


def workspace_agreement(classified: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Accuracy, Cohen's kappa and confusion matrix for the rows a human has labelled."""
    lab = labels.dropna(subset=["manual_theme"])
    lab = lab[lab["manual_theme"].astype(str).str.strip() != ""]
    m = lab.merge(classified[["review_id", "theme", "severity"]].drop_duplicates("review_id"), on="review_id")
    if len(m) < 5:
        return {"n": len(m)}
    sev = m.dropna(subset=["manual_severity"])
    labels_ = sorted(set(m["manual_theme"]) | set(m["theme"]))
    return {
        "n": len(m),
        "theme_accuracy": accuracy_score(m["manual_theme"], m["theme"]),
        "theme_kappa": cohen_kappa_score(m["manual_theme"], m["theme"]),
        "severity_accuracy": accuracy_score(sev["manual_severity"], sev["severity"]) if len(sev) else np.nan,
        "severity_kappa": cohen_kappa_score(sev["manual_severity"], sev["severity"]) if len(sev) > 1 else np.nan,
        "labels": labels_,
        "confusion": confusion_matrix(m["manual_theme"], m["theme"], labels=labels_),
        "disagreements": m[m["manual_theme"] != m["theme"]][["app_name", "review_text", "manual_theme", "theme"]],
    }
