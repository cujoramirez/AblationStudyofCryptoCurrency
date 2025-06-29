# Bitcoin Policy Ablation Study: Comprehensive Analysis

## Executive Summary

This report analyzes the impact of different cryptocurrency policy instruments on 
trading volumes across the United States, Russia, and Indonesia. The analysis is based 
on an ablation study, where each policy instrument was removed to observe the resulting 
percentage change in trading volume compared to the baseline scenario.

> **Note on interpretation:** Positive values indicate that removing the policy 
> instrument would increase trading volume (i.e., the policy was restrictive). 
> Negative values indicate that removing the policy would decrease volume 
> (i.e., the policy was supportive to trading volume).

## Ablation Study Results

The following table shows the percentage change in trading volume when each policy 
instrument is removed from the regulatory framework:

| Policy Instrument              |   Indonesia |   Russia |   United States |
|:-------------------------------|------------:|---------:|----------------:|
| AML Requirements               |        0.89 |     0.42 |            0.00 |
| AML Enforcement                |       -0.15 |     7.98 |           71.45 |
| Digital Financial Asset Class. |      nan    |    -0.20 |          nan    |
| Commodity Classification       |       -0.19 |    -2.99 |           -1.89 |
| Security Classification        |      nan    |   nan    |          -11.61 |
| Exchange Licensing Req.        |        0.20 |    -0.14 |            0.00 |
| License Stringency             |       -0.43 |     1.51 |          -10.64 |
| Payment Ban                    |      nan    |     2.49 |          nan    |
| Regulations                    |      -47.55 |    -6.27 |           26.38 |
| Taxation                       |      -46.90 |     0.00 |          -26.36 |

## Key Findings

### Overall Impact of Regulations

Effects of complete removal of regulations on trading volume

**Indonesia:** Removing all regulations would decrease volume by 47.55%

**Russia:** Removing all regulations would decrease volume by 6.27%

**United States:** Removing all regulations would increase volume by 26.38%



### Most Impactful Policies

Policies with the highest absolute impact on trading volume

**Indonesia:** Regulations (-47.55%)

**Russia:** AML Enforcement (7.98%)

**United States:** AML Enforcement (71.45%)



### Policy Category Analysis

Impact of different policy categories on trading volume

**Indonesia:**

- Classification policies are supportive (-0.19%)

- Licensing policies are supportive (-0.11%)

- Compliance policies are restrictive (0.37%)

- Financial policies are supportive (-46.90%)

- Overall policies are supportive (-47.55%)

**Russia:**

- Classification policies are supportive (-1.60%)

- Licensing policies are restrictive (0.69%)

- Compliance policies are restrictive (4.20%)

- Restrictions policies are restrictive (2.49%)

- Financial policies are supportive (0.00%)

- Overall policies are supportive (-6.27%)

**United States:**

- Classification policies are supportive (-6.75%)

- Licensing policies are supportive (-5.32%)

- Compliance policies are restrictive (35.73%)

- Financial policies are supportive (-26.36%)

- Overall policies are restrictive (26.38%)



### Cross-Country Policy Response

How countries differ in their response to similar policies

**AML Enforcement:** Indonesia: -0.15%, Russia: 7.98%, United States: 71.45%

**AML Requirements:** Indonesia: 0.89%, Russia: 0.42%, United States: 0.00%

**Commodity Classification:** Indonesia: -0.19%, Russia: -2.99%, United States: -1.89%

**Exchange Licensing Req.:** Indonesia: 0.20%, Russia: -0.14%, United States: 0.00%

**License Stringency:** Indonesia: -0.43%, Russia: 1.51%, United States: -10.64%

**Regulations:** Indonesia: -47.55%, Russia: -6.27%, United States: 26.38%

**Taxation:** Indonesia: -46.90%, Russia: 0.00%, United States: -26.36%



## Statistical Summary

### Summary Statistics by Country

| Country       | Mean    | Median   | Std Dev   | Min     | Max    | Most Restrictive   | Most Supportive   |
|:--------------|:--------|:---------|:----------|:--------|:-------|:-------------------|:------------------|
| Indonesia     | -13.45% | -0.19%   | 23.08%    | -47.55% | 0.89%  | AML Requirements   | Regulations       |
| Russia        | 0.31%   | 0.00%    | 3.87%     | -6.27%  | 7.98%  | AML Enforcement    | Regulations       |
| United States | 5.92%   | -0.94%   | 30.40%    | -26.36% | 71.45% | AML Enforcement    | Taxation          |

### Policy Category Impact Analysis

Average percentage change in volume when policies in each category are removed:

| Category       | Indonesia   | Russia   | United States   |
|:---------------|:------------|:---------|:----------------|
| Classification | -0.19%      | -1.60%   | -6.75%          |
| Licensing      | -0.11%      | 0.69%    | -5.32%          |
| Compliance     | 0.37%       | 4.20%    | 35.73%          |
| Restrictions   | N/A         | 2.49%    | N/A             |
| Financial      | -46.90%     | 0.00%    | -26.36%         |
| Overall        | -47.55%     | -6.27%   | 26.38%          |

## Policy Recommendations

### Policies to Maintain or Strengthen

The following policies appear to support cryptocurrency trading volume and could be maintained or strengthened:

**Indonesia:**

- Regulations: Removing this policy would reduce volume by 47.55%

- Taxation: Removing this policy would reduce volume by 46.90%

- License Stringency: Removing this policy would reduce volume by 0.43%

**Russia:**

- Regulations: Removing this policy would reduce volume by 6.27%

- Commodity Classification: Removing this policy would reduce volume by 2.99%

- Digital Financial Asset Class.: Removing this policy would reduce volume by 0.20%

**United States:**

- Taxation: Removing this policy would reduce volume by 26.36%

- Security Classification: Removing this policy would reduce volume by 11.61%

- License Stringency: Removing this policy would reduce volume by 10.64%

### Policies to Reconsider or Refine

The following policies appear to significantly restrict cryptocurrency trading volume and might be candidates for refinement:

**Indonesia:**

- AML Requirements: Removing this policy would increase volume by 0.89%

- Exchange Licensing Req.: Removing this policy would increase volume by 0.20%

**Russia:**

- AML Enforcement: Removing this policy would increase volume by 7.98%

- Payment Ban: Removing this policy would increase volume by 2.49%

- License Stringency: Removing this policy would increase volume by 1.51%

**United States:**

- AML Enforcement: Removing this policy would increase volume by 71.45%

- Regulations: Removing this policy would increase volume by 26.38%

## Conclusion

This ablation study reveals significant differences in how cryptocurrency policy instruments 
affect trading volumes across different countries. Some key observations include:

- United States policies are overall restrictive to trading volume (avg: 5.92%)

- Russian policies are overall restrictive to trading volume (avg: 0.31%)

- Indonesian policies are overall supportive to trading volume (avg: -13.45%)


Policy makers should consider these findings when crafting cryptocurrency regulations, 
particularly noting the different impacts of similar policies across jurisdictions, which suggests 
that local market conditions and complementary policies play important roles in determining outcomes.
