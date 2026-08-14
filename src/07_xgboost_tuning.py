"""
7단계: XGBoost 하이퍼파라미터 튜닝 (GridSearchCV, 5-fold CV)
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
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
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(base_model, param_grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, verbose=0)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

lines = ['# XGBoost 하이퍼파라미터 튜닝 결과\n']
lines.append('GridSearchCV, 5-fold 교차검증, scoring=neg_root_mean_squared_error\n')
lines.append(f'\n## 최적 하이퍼파라미터\n{grid.best_params_}\n')
lines.append(f'\nCV 최적 RMSE: {-grid.best_score_:.2f}\n')
lines.append(f'\n## test set 성능\n- RMSE: {rmse:.2f}\n- MAE: {mae:.2f}\n- R2: {r2:.4f}\n')
lines.append('\n## 변수 중요도 (튜닝 후)\n| 변수 | 중요도 |\n|---|---|\n')
for c, imp in sorted(zip(feature_cols, best_model.feature_importances_), key=lambda x: -x[1]):
    lines.append(f'| {c} | {imp:.4f} |\n')

with open('outputs/model/xgboost_tuning_result.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/model/xgboost_tuning_result.md')
print('최적 하이퍼파라미터:', grid.best_params_)
print(f'RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}')
