# Ridge / Lasso 회귀 결과 (train=2015, test=2016)
alpha 탐색: RidgeCV/LassoCV + 5-fold TimeSeriesSplit(train 2015 내부에서만), 표준화된 입력 사용

## Ridge
- 선택된 alpha: 0.7318
- RMSE: 4357.40
- MAE: 3385.77
- R2: 0.7501

### 표준화 회귀계수
| 변수 | 계수 |
|---|---|
| t2 | 4195.09 |
| hum | -3158.96 |
| workingday | 2474.90 |
| wind_speed | -2221.08 |
| weather_code | -1166.71 |
| mnth | -486.17 |
| weekday | 401.09 |
| is_holiday | -270.79 |
| season | -86.64 |

## Lasso
- 선택된 alpha: 16.6088
- RMSE: 4351.50
- MAE: 3381.41
- R2: 0.7508
- 계수가 0으로 수렴한 변수 수: 0 / 9

### 표준화 회귀계수
| 변수 | 계수 |
|---|---|
| t2 | 4177.58 |
| hum | -3167.54 |
| workingday | 2465.80 |
| wind_speed | -2210.59 |
| weather_code | -1157.93 |
| mnth | -463.50 |
| weekday | 386.11 |
| is_holiday | -256.95 |
| season | -77.38 |
