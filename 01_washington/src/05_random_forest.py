"""
5단계: 랜덤포레스트 회귀
casual/registered를 제외한 11개 변수로 학습하고 성능/중요도를 저장한다.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from common import get_feature_cols

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

feature_cols = get_feature_cols(train_df)
X_train, y_train = train_df[feature_cols], train_df['cnt']
X_test, y_test = test_df[feature_cols], test_df['cnt']

model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

lines = ['# 랜덤포레스트 회귀 결과\n']
lines.append('RandomForestRegressor(n_estimators=300, random_state=42)\n')
lines.append(f'\n- RMSE: {rmse:.2f}\n- MAE: {mae:.2f}\n- R2: {r2:.4f}\n')
lines.append('\n## 변수 중요도\n| 변수 | 중요도 |\n|---|---|\n')
for c, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
    lines.append(f'| {c} | {imp:.4f} |\n')

with open('outputs/model/random_forest_result.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/model/random_forest_result.md')
print(f'RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}')
