**## TL;DR**  
- The 5‑day window captured **300 reviews** with an overall **average rating of 3.17**.  
- **Zomato** outperforms peers with an **average rating of 4.17** (vs Swiggy 3.68 and Swiggy iOS 1.67).  
- **Late Delivery** appears in **7.67 %** of reviews and drives the largest rating‑drag (**‑0.18** points).

---  

**## Competitive Position**  

- **Zomato leads** overall satisfaction (4.17 ★) and has the lowest high‑severity issue rate (**2 %**).  
- **Swiggy (iOS) lags** dramatically with a 1.67 ★ rating, **33 %** high‑severity issues and the highest issue share (**72 %**).  
- **Swiggy (iOS) shows a *significant* gap** on **Late Delivery** (18 % of its issues vs 2.5 % peer average).  
- **Zomato shows a *significant* advantage** on **Refund / Cancellation** (0 % of its issues vs 7 % peer average).  

---  

**## Biggest Risks**  

| Risk | What users say (verbatim) | Evidence (numbers) | Likely root‑cause hypothesis |
|------|---------------------------|--------------------|------------------------------|
| **Late Delivery** | “*ye log ne mera order liya … 1 hour … agent se connect he nhi kar pa raha…*” (Swiggy)  <br> “*Not delivering the food on time and not responding to my call. Worst service*” (Swiggy iOS) | Share = 7.67 % of reviews, **high‑severity = 47.83 %**, priority score = 62, rating‑drag = ‑0.18. Swiggy iOS top‑issue = Late Delivery (18 % of its issues). | Partner availability & dispatch routing bottlenecks; insufficient real‑time ETA updates. |
| **Refund / Cancellation** | “*An excellent lesson in how not to handle customer service! … Multiple complaints, evidence*” (Swiggy)  <br> “*Pathetic Customer Service! Swiggy cancelled my order…*” (Swiggy) | Share = 4.67 % of reviews, **high‑severity = 100 %**, priority = 55.6, rating‑drag = ‑0.10. | Order‑status synchronization failures; opaque cancellation policies causing escalations. |
| **Customer Support** | “*Clumsy customer service. I strongly recommend avoiding Swiggy…*” (Swiggy iOS)  <br> “*Pathetic customer service. Almost looting people.*” (Swiggy iOS) | Share = 5 % of reviews, **high‑severity = 33.33 %**, rating‑drag = ‑0.11, flagged as a *quick win*. | Undersized support team, lack of escalation matrix, poor agent training. |

---  

**## Recommended Actions (next 30‑60‑90 days)**  

| Horizon | Action | Owner (team) | Success metric |
|---------|--------|--------------|----------------|
| **30 days** | Deploy partner‑performance dashboard to flag deliveries >30 min late; send targeted alerts to offending partners. | Ops / Analytics | ↓ Late‑Delivery high‑severity % from 47.8 % to ≤30 % |
| **60 days** | Redesign the refund/cancellation flow: add instant in‑app status, auto‑refund triggers for out‑of‑stock cancellations. | Product / Engineering | ↓ Refund/Cancellation high‑severity % from 100 % to ≤20 % |
| **90 days** | Launch AI‑assisted support chat & up‑skill agents on escalation handling. | Customer Support | ↓ Customer‑Support high‑severity % from 33.3 % to ≤10 %; improve NPS proxy by +5 points |

---  

**## Opportunities**  

- **Add Cash‑on‑Delivery** – requested by both Swiggy (1) and Zomato (1) users; could capture price‑sensitive segment.  
- **Show Cashback Offers** – a Zomato‑only request; integrating dynamic offer banners may boost conversion.  
- **Leverage Zomato’s weak Refund/Cancellation performance** (0 % issues vs 7 % peer) by promoting Swiggy’s faster, transparent refund process once improved.

---  

**## Caveats**  

- The analysis covers a **5‑day review window (15‑19 Nov 2026)**, which may not reflect longer‑term trends.  
- Themes and sentiment are derived via **LLM classification**; occasional mis‑tagging can affect share percentages.