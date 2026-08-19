"""
7단계: XGBoost 기본값 (train=2015 -> test=2016)
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from common import get_feature_cols, TARGET_COL

train_df = pd.read_csv('data/processed/day_london_2015.csv')
test_df = pd.read_csv('data/processed/day_london_2016.csv')

feature_cols = get_feature_cols()
X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

model = XGBRegressor(random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

lines = ['# XGBoost(기본값) 분석 결과 (train=2015, test=2016)\n']
lines.append(f'\n- RMSE: {rmse:.2f}\n- MAE: {mae:.2f}\n- R2: {r2:.4f}\n')
lines.append('\n## 변수 중요도\n| 변수 | 중요도 |\n|---|---|\n')
for c, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
    lines.append(f'| {c} | {imp:.4f} |\n')

with open('outputs/model/xgboost_baseline_result.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/model/xgboost_baseline_result.md')
print(f'RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}')
