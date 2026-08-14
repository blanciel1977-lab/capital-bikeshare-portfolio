# XGBoost 회귀 결과 (기본 하이퍼파라미터)
XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)

- RMSE: 980.80
- MAE: 804.28
- R2: 0.7263

## 변수 중요도
| 변수 | 중요도 |
|---|---|
| yr | 0.3887 |
| temp | 0.2024 |
| atemp | 0.1543 |
| season | 0.0957 |
| weathersit | 0.0669 |
| hum | 0.0241 |
| mnth | 0.0231 |
| windspeed | 0.0172 |
| holiday | 0.0104 |
| workingday | 0.0101 |
| weekday | 0.0073 |
