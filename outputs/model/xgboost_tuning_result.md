# XGBoost 하이퍼파라미터 튜닝 결과
GridSearchCV, 5-fold 교차검증, scoring=neg_root_mean_squared_error

## 최적 하이퍼파라미터
{'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 300, 'subsample': 0.8}

CV 최적 RMSE: 588.27

## test set 성능
- RMSE: 958.33
- MAE: 774.09
- R2: 0.7387

## 변수 중요도 (튜닝 후)
| 변수 | 중요도 |
|---|---|
| yr | 0.4071 |
| atemp | 0.2066 |
| temp | 0.1470 |
| season | 0.0758 |
| weathersit | 0.0647 |
| hum | 0.0259 |
| mnth | 0.0200 |
| windspeed | 0.0169 |
| workingday | 0.0127 |
| holiday | 0.0122 |
| weekday | 0.0110 |
