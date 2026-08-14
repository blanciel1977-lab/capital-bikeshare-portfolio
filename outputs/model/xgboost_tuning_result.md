# XGBoost 하이퍼파라미터 튜닝 결과
GridSearchCV, 5-fold **TimeSeriesSplit** 교차검증, scoring=neg_root_mean_squared_error

## 최적 하이퍼파라미터
{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.8}

CV 최적 RMSE: 1154.46

## test set 성능
- RMSE: 968.75
- MAE: 770.37
- R2: 0.7329

## 교차검증 방식에 따른 점수 차이 (동일 하이퍼파라미터 기준)

| 채점 방식 | RMSE | 실제(test)와의 격차 |
|---|---|---|
| 무작위 KFold (기존 방식) | 604.56 | +364 |
| TimeSeriesSplit (현재 채택) | 1154.46 | -186 |
| **실제 test set** | **968.75** | — |

무작위 KFold는 검증 조각의 앞뒤 날짜가 학습 조각에 포함되어 실제보다 후한 점수를 준다. TimeSeriesSplit은 항상 과거로 학습해 미래를 검증하므로 실제 test 성능에 더 가까운 값을 내며, 따라서 하이퍼파라미터도 실제 사용 시나리오와 같은 기준으로 선택된다.

## 변수 중요도 (튜닝 후)
| 변수 | 중요도 |
|---|---|
| yr | 0.4597 |
| atemp | 0.1942 |
| weathersit | 0.1187 |
| season | 0.1085 |
| mnth | 0.0283 |
| hum | 0.0245 |
| workingday | 0.0202 |
| windspeed | 0.0166 |
| holiday | 0.0159 |
| weekday | 0.0134 |
