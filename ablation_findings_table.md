# Ablation Findings by Policy Instrument

This table shows the percentage change in trading volume when a policy instrument is removed from the regulatory framework. The analysis is based on an XGBoost model trained on historical crypto trading data across three countries.

| Policy Instrument | Indonesia | Russia | United States |
|:--------------------:|:---:|:---:|:---:|
| Regulations | -47.55% | -6.27% | +26.38% |
| AML Enforcement | -0.15% | +7.98% | +71.45% |
| Taxation | -46.90% | 0.00% | -26.36% |
| Security Classification | N/A | N/A | -11.61% |
| License Stringency | -0.43% | +1.51% | -10.64% |
| Payment Ban | N/A | +2.49% | N/A |
| Commodity Classification | -0.19% | -2.99% | -1.89% |
| AML Requirements | +0.89% | +0.42% | 0.00% |
| Digital Financial Asset Class. | N/A | -0.20% | N/A |
| Exchange Licensing Req. | +0.20% | -0.14% | 0.00% |
## Understanding the Table
- **Positive values (+)**: Removing this policy *increases* trading volume, suggesting the policy had a negative impact on volume.
- **Negative values (-)**: Removing this policy *decreases* trading volume, suggesting the policy had a positive impact on volume.
- **N/A**: This policy was not implemented or simulated in the country.

## Key Insights
1. **AML Enforcement** has the most significant positive impact on US volumes (+71.45%), suggesting strong enforcement increases compliance and market confidence.
2. **Security Classification** has a significant negative impact in the US (-11.61%), indicating this regulatory classification supports higher trading volumes.
3. **Taxation** has substantial effects on Indonesia (-46.90%) and US (-26.36%), with both countries showing decreased volumes when tax policies are removed.
4. **Complete removal of regulations** produces mixed effects: decreased volume in Indonesia (-47.55%) and Russia (-6.27%), but increased volume in the US (+26.38%).
