# XGBoost 하이퍼파라미터 튜닝 결과 (train=2015, test=2016)
GridSearchCV, 5-fold **TimeSeriesSplit** 교차검증(train 2015 내부에서만), scoring=neg_root_mean_squared_error

## 최적 하이퍼파라미터
{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300, 'subsample': 1.0}

CV 최적 RMSE: 5583.98

## test(2016) set 성능
- RMSE: 4419.45
- MAE: 3148.64
- R2: 0.7429

## 교차검증 방식에 따른 점수 차이 (동일 하이퍼파라미터 기준)

| 채점 방식 | RMSE | 실제(test)와의 격차 |
|---|---|---|
| 무작위 KFold (참고용) | 4502.72 | -83 |
| TimeSeriesSplit (채택) | 5583.98 | -1165 |
| **실제 test(2016)** | **4419.45** | — |

## 변수 중요도 (튜닝 후)
| 변수 | 중요도 |
|---|---|
| workingday | 0.3199 |
| season | 0.1857 |
| t2 | 0.1645 |
| weather_code | 0.1610 |
| hum | 0.0853 |
| mnth | 0.0324 |
| wind_speed | 0.0218 |
| weekday | 0.0204 |
| is_holiday | 0.0090 |
