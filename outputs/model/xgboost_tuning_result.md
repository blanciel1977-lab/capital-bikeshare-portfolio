# XGBoost 하이퍼파라미터 튜닝 결과
GridSearchCV, 5-fold 교차검증, scoring=neg_root_mean_squared_error

## 최적 하이퍼파라미터
{'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.8}

CV 최적 RMSE: 650.99

## test set 성능
- RMSE: 626.23
- MAE: 428.57
- R2: 0.9022

## 변수 중요도 (튜닝 후)
| 변수 | 중요도 |
|---|---|
| yr | 0.3395 |
| season | 0.1660 |
| atemp | 0.1612 |
| temp | 0.1514 |
| weathersit | 0.0622 |
| hum | 0.0292 |
| mnth | 0.0201 |
| holiday | 0.0191 |
| windspeed | 0.0184 |
| weekday | 0.0175 |
| workingday | 0.0155 |
