"""
7단계: XGBoost 하이퍼파라미터 튜닝 (GridSearchCV, 5-fold TimeSeriesSplit)

교차검증 분할도 시간 순서를 지켜야 한다.
기본 KFold(shuffle=True)는 학습 구간 내부를 무작위로 섞으므로, 3단계에서
train/test를 시간순으로 나눈 의미가 튜닝 단계에서 다시 사라진다. 검증 조각의
앞뒤 날짜가 학습 조각에 포함되어 '양옆 값을 보고 가운데를 메우는' 쉬운 문제가
되고, 그 결과 CV 점수가 낙관적으로 나올 뿐 아니라 **그 쉬운 문제에 유리한
하이퍼파라미터**가 선택된다(예: 국소 패턴을 외우기 좋은 더 깊은 트리).

TimeSeriesSplit은 항상 '앞 구간으로 학습 -> 바로 다음 구간으로 검증'하도록
분할을 확장해가므로, 실제 사용 시나리오(과거로 배워 미래를 예측)와 같은
기준으로 하이퍼파라미터를 고르게 된다.

※ TimeSeriesSplit은 행 순서를 시간 순으로 가정하므로 data/train.csv가
   날짜순으로 정렬되어 있어야 한다(3단계에서 보장).
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

feature_cols = [c for c in train_df.columns if c not in ['instant', 'dteday', 'cnt', 'casual', 'registered']]
X_train, y_train = train_df[feature_cols], train_df['cnt']
X_test, y_test = test_df[feature_cols], test_df['cnt']

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 4, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
}

base_model = XGBRegressor(random_state=42, n_jobs=-1)
cv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(base_model, param_grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

# 참고: 선택된 설정을 '무작위 KFold'로도 채점해, 기존 방식이 얼마나 낙관적이었는지 보여준다.
kfold_rmse = -cross_val_score(
    XGBRegressor(**{**grid.best_params_, 'random_state': 42, 'n_jobs': -1}),
    X_train, y_train,
    scoring='neg_root_mean_squared_error',
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
).mean()

lines = ['# XGBoost 하이퍼파라미터 튜닝 결과\n']
lines.append('GridSearchCV, 5-fold **TimeSeriesSplit** 교차검증, scoring=neg_root_mean_squared_error\n')
lines.append(f'\n## 최적 하이퍼파라미터\n{grid.best_params_}\n')
lines.append(f'\nCV 최적 RMSE: {-grid.best_score_:.2f}\n')
lines.append(f'\n## test set 성능\n- RMSE: {rmse:.2f}\n- MAE: {mae:.2f}\n- R2: {r2:.4f}\n')

lines.append('\n## 교차검증 방식에 따른 점수 차이 (동일 하이퍼파라미터 기준)\n\n')
lines.append('| 채점 방식 | RMSE | 실제(test)와의 격차 |\n|---|---|---|\n')
lines.append(f'| 무작위 KFold (기존 방식) | {kfold_rmse:.2f} | {rmse - kfold_rmse:+.0f} |\n')
lines.append(f'| TimeSeriesSplit (현재 채택) | {-grid.best_score_:.2f} | {rmse - (-grid.best_score_):+.0f} |\n')
lines.append(f'| **실제 test set** | **{rmse:.2f}** | — |\n')
lines.append('\n무작위 KFold는 검증 조각의 앞뒤 날짜가 학습 조각에 포함되어 실제보다 후한 점수를 준다. '
             'TimeSeriesSplit은 항상 과거로 학습해 미래를 검증하므로 실제 test 성능에 더 가까운 값을 내며, '
             '따라서 하이퍼파라미터도 실제 사용 시나리오와 같은 기준으로 선택된다.\n')
lines.append('\n## 변수 중요도 (튜닝 후)\n| 변수 | 중요도 |\n|---|---|\n')
for c, imp in sorted(zip(feature_cols, best_model.feature_importances_), key=lambda x: -x[1]):
    lines.append(f'| {c} | {imp:.4f} |\n')

with open('outputs/model/xgboost_tuning_result.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/model/xgboost_tuning_result.md')
print('최적 하이퍼파라미터:', grid.best_params_)
print(f'CV(TimeSeriesSplit) RMSE={-grid.best_score_:.2f} / CV(무작위 KFold) RMSE={kfold_rmse:.2f}')
print(f'test: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}')
