"""
6단계: XGBoost 회귀 (기본 하이퍼파라미터)
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

feature_cols = [c for c in train_df.columns if c not in ['instant', 'dteday', 'cnt', 'casual', 'registered']]
X_train, y_train = train_df[feature_cols], train_df['cnt']
X_test, y_test = test_df[feature_cols], test_df['cnt']

model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

lines = ['# XGBoost 회귀 결과 (기본 하이퍼파라미터)\n']
lines.append('XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)\n')
lines.append(f'\n- RMSE: {rmse:.2f}\n- MAE: {mae:.2f}\n- R2: {r2:.4f}\n')
lines.append('\n## 변수 중요도\n| 변수 | 중요도 |\n|---|---|\n')
for c, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
    lines.append(f'| {c} | {imp:.4f} |\n')

with open('outputs/model/xgboost_baseline_result.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/model/xgboost_baseline_result.md')
print(f'RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}')
