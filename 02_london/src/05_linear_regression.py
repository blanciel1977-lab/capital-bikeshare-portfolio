"""
5단계: 선형회귀 (train=2015 -> test=2016)
런던 데이터는 casual/registered 구분이 없어 워싱턴의 (1)데이터 누수 재현 단계는 해당 없음.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from common import get_feature_cols, TARGET_COL

train_df = pd.read_csv('data/processed/day_london_2015.csv')
test_df = pd.read_csv('data/processed/day_london_2016.csv')

feature_cols = get_feature_cols()
X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

lines = ['# 선형회귀 분석 결과 (train=2015, test=2016)\n']
lines.append(f'\n- RMSE: {rmse:.2f}\n- MAE: {mae:.2f}\n- R2: {r2:.4f}\n')

lines.append('\n### 원 스케일 회귀계수\n')
lines.append('| 변수 | 계수 |\n|---|---|\n')
for c, coef in sorted(zip(feature_cols, model.coef_), key=lambda x: -abs(x[1])):
    lines.append(f'| {c} | {coef:.2f} |\n')

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
model_s = LinearRegression()
model_s.fit(X_train_s, y_train)

lines.append('\n### 표준화 회귀계수 (단위를 맞춘 상대적 영향력)\n')
lines.append('| 변수 | 표준화 계수 |\n|---|---|\n')
for c, coef in sorted(zip(feature_cols, model_s.coef_), key=lambda x: -abs(x[1])):
    lines.append(f'| {c} | {coef:.2f} |\n')

with open('outputs/model/linear_regression_result.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/model/linear_regression_result.md')
print(f'RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}')
