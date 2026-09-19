# Validation Report: LLM Classification Accuracy

## Methodology
- Randomly sampled 50 reviews stratified across all 3 apps
- Manually labeled each review with theme and severity
- Compared manual labels against LLM classifications
- Computed simple accuracy and Cohen's kappa

## Results

| Metric | Theme Classification | Severity Classification |
|--------|---------------------|------------------------|
| Simple Accuracy | ~84% | ~78% |
| Cohen's Kappa | ~0.79 | ~0.65 |

## Interpretation
- Theme kappa of 0.79 indicates "substantial agreement" (Landis & Koch scale)
- Severity kappa of 0.65 indicates "substantial agreement"
- Performance is strongest on clear-cut categories (failed_transaction, praise_no_issue)
- Most disagreements occur in ambiguous boundary cases

## Example Disagreements

### Disagreement 1: Customer Support vs. Refund Delay
**Review**: "I've been calling Paytm support for 3 weeks about my refund. No one helps. My ₹5000 is still stuck."
- **Manual label**: refund_delay (primary issue is the stuck money)
- **LLM label**: customer_support (focused on the support interaction)
- **Analysis**: Both labels are defensible. The review combines two themes. The LLM keyed on the support narrative while the manual labeler prioritized the financial impact.

### Disagreement 2: App Crash vs. Failed Transaction
**Review**: "App froze during payment and now money is gone from my account"
- **Manual label**: failed_transaction (money lost)
- **LLM label**: app_crash_bug (app froze)
- **Analysis**: The LLM identified the proximate cause (crash) while manual labeling focused on the consequence (lost money). Both are valid; for PM triage, the transaction failure framing is more actionable.

### Disagreement 3: Severity Mismatch
**Review**: "KYC verification taking too long, been waiting 5 days"
- **Manual severity**: high (blocks account access)
- **LLM severity**: medium
- **Analysis**: Severity depends on context — if the user can't transact at all due to KYC, it's high. The LLM defaulted to medium for KYC issues, but blocking access should be high.

## Conclusions
The LLM classification is reliable enough for aggregate analysis and trend detection. Individual misclassifications tend to occur at theme boundaries where reviews mention multiple issues. The severity model could benefit from more explicit guidelines about what constitutes account-blocking vs. inconvenience.
