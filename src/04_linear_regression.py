"""
4단계: 선형회귀 분석
- (1) casual/registered를 포함해 학습 시 발생하는 데이터 누수(data leakage)를 재현하여 원인을 보여줌
- (2) casual/registered를 제외한 11개 변수로 올바르게 재학습
결과를 outputs/model/linear_regression_result.md에 저장한다.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from common import get_feature_cols

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

lines = ['# 선형회귀 분석 결과\n']

# --- (1) 데이터 누수 재현 ---
leak_cols = [c for c in train_df.columns if c not in ['instant', 'dteday', 'cnt']]
X_train, y_train = train_df[leak_cols], train_df['cnt']
X_test, y_test = test_df[leak_cols], test_df['cnt']

model_leak = LinearRegression()
model_leak.fit(X_train, y_train)
pred_leak = model_leak.predict(X_test)
rmse_leak = np.sqrt(mean_squared_error(y_test, pred_leak))
r2_leak = r2_score(y_test, pred_leak)

lines.append('## (1) 데이터 누수 재현: casual/registered를 입력에 포함한 경우\n')
lines.append(f'- RMSE: {rmse_leak:.4f}, R2: {r2_leak:.4f}\n')
lines.append('- cnt = casual + registered로 정의된 값이라, 두 컬럼을 입력에 넣으면 모델이 '
              '항등식을 그대로 학습해 R2가 사실상 1.0이 된다 (leakage).\n')
lines.append(f"- casual 계수: {dict(zip(leak_cols, model_leak.coef_)).get('casual'):.4f}, "
              f"registered 계수: {dict(zip(leak_cols, model_leak.coef_)).get('registered'):.4f}\n")

# --- (2) casual/registered 제외 후 재학습 ---
feature_cols = get_feature_cols(train_df)
X_train, y_train = train_df[feature_cols], train_df['cnt']
X_test, y_test = test_df[feature_cols], test_df['cnt']

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

lines.append('\n## (2) casual/registered 제외 후 재학습 (11개 변수)\n')
lines.append(f'- RMSE: {rmse:.2f}\n- MAE: {mae:.2f}\n- R2: {r2:.4f}\n')

lines.append('\n### 원 스케일 회귀계수\n')
lines.append('| 변수 | 계수 |\n|---|---|\n')
for c, coef in sorted(zip(feature_cols, model.coef_), key=lambda x: -abs(x[1])):
    lines.append(f'| {c} | {coef:.2f} |\n')

# --- 표준화 회귀계수 (영향력 비교용) ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
model_s = LinearRegression()
model_s.fit(X_train_s, y_train)
pred_s = model_s.predict(X_test_s)

lines.append('\n### 표준화 회귀계수 (단위를 맞춘 상대적 영향력)\n')
lines.append('| 변수 | 표준화 계수 |\n|---|---|\n')
for c, coef in sorted(zip(feature_cols, model_s.coef_), key=lambda x: -abs(x[1])):
    lines.append(f'| {c} | {coef:.2f} |\n')

with open('outputs/model/linear_regression_result.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/model/linear_regression_result.md')
print(f'(누수 포함) RMSE={rmse_leak:.4f}, R2={r2_leak:.4f}')
print(f'(정상 학습) RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}')
