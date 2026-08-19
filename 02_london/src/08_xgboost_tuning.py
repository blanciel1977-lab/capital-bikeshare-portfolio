"""
8단계: XGBoost 하이퍼파라미터 튜닝 (GridSearchCV, 5-fold TimeSeriesSplit, train=2015 내부에서만)
그리드는 워싱턴과 동일 범위 유지 (사용자 확정).
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from common import get_feature_cols, TARGET_COL

train_df = pd.read_csv('data/processed/day_london_2015.csv')
test_df = pd.read_csv('data/processed/day_london_2016.csv')

feature_cols = get_feature_cols()
X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

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

kfold_rmse = -cross_val_score(
    XGBRegressor(**{**grid.best_params_, 'random_state': 42, 'n_jobs': -1}),
    X_train, y_train,
    scoring='neg_root_mean_squared_error',
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
).mean()

lines = ['# XGBoost 하이퍼파라미터 튜닝 결과 (train=2015, test=2016)\n']
lines.append('GridSearchCV, 5-fold **TimeSeriesSplit** 교차검증(train 2015 내부에서만), scoring=neg_root_mean_squared_error\n')
lines.append(f'\n## 최적 하이퍼파라미터\n{grid.best_params_}\n')
lines.append(f'\nCV 최적 RMSE: {-grid.best_score_:.2f}\n')
lines.append(f'\n## test(2016) set 성능\n- RMSE: {rmse:.2f}\n- MAE: {mae:.2f}\n- R2: {r2:.4f}\n')

lines.append('\n## 교차검증 방식에 따른 점수 차이 (동일 하이퍼파라미터 기준)\n\n')
lines.append('| 채점 방식 | RMSE | 실제(test)와의 격차 |\n|---|---|---|\n')
lines.append(f'| 무작위 KFold (참고용) | {kfold_rmse:.2f} | {rmse - kfold_rmse:+.0f} |\n')
lines.append(f'| TimeSeriesSplit (채택) | {-grid.best_score_:.2f} | {rmse - (-grid.best_score_):+.0f} |\n')
lines.append(f'| **실제 test(2016)** | **{rmse:.2f}** | — |\n')

lines.append('\n## 변수 중요도 (튜닝 후)\n| 변수 | 중요도 |\n|---|---|\n')
for c, imp in sorted(zip(feature_cols, best_model.feature_importances_), key=lambda x: -x[1]):
    lines.append(f'| {c} | {imp:.4f} |\n')

with open('outputs/model/xgboost_tuning_result.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/model/xgboost_tuning_result.md')
print('최적 하이퍼파라미터:', grid.best_params_)
print(f'CV(TimeSeriesSplit) RMSE={-grid.best_score_:.2f} / CV(무작위 KFold) RMSE={kfold_rmse:.2f}')
print(f'test(2016): RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}')
