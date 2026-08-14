# XGBoost 회귀 결과 (기본 하이퍼파라미터)
XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)

- RMSE: 655.49
- MAE: 446.11
- R2: 0.8928

## 변수 중요도
| 변수 | 중요도 |
|---|---|
| yr | 0.4467 |
| temp | 0.1968 |
| atemp | 0.1002 |
| season | 0.0901 |
| weathersit | 0.0589 |
| hum | 0.0255 |
| mnth | 0.0222 |
| holiday | 0.0211 |
| windspeed | 0.0153 |
| workingday | 0.0132 |
| weekday | 0.0101 |
