# XGBoost 회귀 결과 (기본 하이퍼파라미터)
XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)

- RMSE: 1040.61
- MAE: 850.31
- R2: 0.6919

## 변수 중요도
| 변수 | 중요도 |
|---|---|
| yr | 0.5064 |
| atemp | 0.2214 |
| season | 0.1024 |
| weathersit | 0.0670 |
| hum | 0.0278 |
| mnth | 0.0244 |
| windspeed | 0.0175 |
| holiday | 0.0133 |
| workingday | 0.0111 |
| weekday | 0.0087 |
