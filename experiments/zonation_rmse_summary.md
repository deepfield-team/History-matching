# SPE1 RMSE Summary

Rates are aggregated over production wells; BHP is averaged across prod/inj wells.
BHP RMSE is reported in bar.

| Technique | RMSE ORAT (prod) | RMSE GRAT (prod) | RMSE WRAT (prod) | RMSE BHP (prod) | RMSE BHP (inj) | Perm RMSE L1 (mD) | Perm RMSE L2 (mD) | Perm RMSE L3 (mD) |
|---|---|---|---|---|---|---|---|---|
| sign_uncons | 0.000243 | 0.798 | 5.226e-12 | 4.76 | 3.279 | 211.3 | 35.77 | 52.51 |
| sign_cons | 0.0001918 | 0.9713 | 1.356e-11 | 5.463 | 3.152 | 255.4 | 48.99 | 50.37 |
| medium-cons | 0.0002218 | 0.8735 | 7.363e-12 | 5.251 | 2.937 | 165.5 | 69.95 | 233.2 |
| medium_uncons | 0.000226 | 0.8851 | 5.555e-12 | 5.212 | 2.701 | 216.3 | 61.23 | 171.5 |
| hierarchical clustering | 0.01321 | 20.25 | 6.866e-11 | 106.2 | 187.1 | 346 | 12.9 | 34.29 |
| no_zonation | 0.002194 | 4.313 | 4.71e-11 | 8.022 | 22.31 | 477.2 | 143.9 | 571.3 |
