"""
16단계: season/weather_code 원-핫 인코딩 + Ridge 재실험

jskim414/london-bike-demand-analysis(docs/good_templet.md 참고)에서 season/weather_code를
원-핫 인코딩한 뒤 Ridge가 랜덤포레스트를 이겼다는 결과가 나왔다. 우리 데이터로도 같은
효과가 재현되는지 확인한다.

- season, weather_code: 원-핫 인코딩 (train 기준으로 학습, test는 handle_unknown='ignore')
- weather_code=26(눈)은 train에 단 1일, test에 2일뿐인 희귀 범주 — 무규제 회귀에서
  계수가 불안정해지는지, Ridge가 이를 안정화하는지 직접 확인한다.
- 나머지 연속형(t2, hum, wind_speed)과 이진/서수(weekday, mnth, is_holiday, workingday)는
  기존과 동일하게 사용, 연속형만 표준화한다.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from common import TARGET_COL

train_df = pd.read_csv('data/processed/day_london_2015.csv')
test_df = pd.read_csv('data/processed/day_london_2016.csv')

ONEHOT_COLS = ['season', 'weather_code']
NUMERIC_COLS = ['t2', 'hum', 'wind_speed']
PASSTHROUGH_COLS = ['mnth', 'weekday', 'is_holiday', 'workingday']

feature_cols = ONEHOT_COLS + NUMERIC_COLS + PASSTHROUGH_COLS
X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

pre = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), ONEHOT_COLS),
    ('scale', StandardScaler(), NUMERIC_COLS),
    ('passthrough', 'passthrough', PASSTHROUGH_COLS),
])
X_train_enc = pre.fit_transform(X_train)
X_test_enc = pre.transform(X_test)

onehot_names = list(pre.named_transformers_['onehot'].get_feature_names_out(ONEHOT_COLS))
all_names = onehot_names + NUMERIC_COLS + PASSTHROUGH_COLS

lines = ['# 원-핫 인코딩 + Ridge 재실험 (train=2015, test=2016)\n\n']
lines.append(f'`season`, `weather_code`를 원-핫 인코딩({len(onehot_names)}개 더미: {", ".join(onehot_names)}), '
             f'`t2/hum/wind_speed`는 표준화, 나머지는 그대로 사용.\n')
lines.append(f'\nweather_code=26.0 표본 수: train {int((train_df["weather_code"]==26.0).sum())}일, '
             f'test {int((test_df["weather_code"]==26.0).sum())}일 (극희귀 범주)\n')

# --- (1) 무규제 선형회귀: 희귀 범주 계수 불안정성 확인 ---
lr = LinearRegression()
lr.fit(X_train_enc, y_train)
pred_lr = lr.predict(X_test_enc)
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
mae_lr = mean_absolute_error(y_test, pred_lr)
r2_lr = r2_score(y_test, pred_lr)

coef_lr = dict(zip(all_names, lr.coef_))
wc26_key = [n for n in all_names if 'weather_code' in n and '26' in n]
wc26_coef_lr = coef_lr.get(wc26_key[0]) if wc26_key else None

lines.append('\n## (1) 무규제 선형회귀 (원-핫 인코딩)\n')
lines.append(f'- RMSE: {rmse_lr:.2f}\n- MAE: {mae_lr:.2f}\n- R2: {r2_lr:.4f}\n')
if wc26_coef_lr is not None:
    lines.append(f'- `{wc26_key[0]}` 계수: **{wc26_coef_lr:+.1f}** (표본 1일뿐인 희귀 범주 — 불안정 여부 확인용)\n')

# --- (2) Ridge: 정규화로 안정화되는지 확인 ---
cv = TimeSeriesSplit(n_splits=5)
alphas = np.logspace(-2, 3, 60)
ridge = RidgeCV(alphas=alphas, cv=cv)
ridge.fit(X_train_enc, y_train)
pred_ridge = ridge.predict(X_test_enc)
rmse_ridge = np.sqrt(mean_squared_error(y_test, pred_ridge))
mae_ridge = mean_absolute_error(y_test, pred_ridge)
r2_ridge = r2_score(y_test, pred_ridge)

coef_ridge = dict(zip(all_names, ridge.coef_))
wc26_coef_ridge = coef_ridge.get(wc26_key[0]) if wc26_key else None

lines.append('\n## (2) Ridge (원-핫 인코딩, TimeSeriesSplit으로 alpha 탐색)\n')
lines.append(f'- 선택된 alpha: {ridge.alpha_:.4f}\n')
lines.append(f'- RMSE: {rmse_ridge:.2f}\n- MAE: {mae_ridge:.2f}\n- R2: {r2_ridge:.4f}\n')
if wc26_coef_ridge is not None:
    lines.append(f'- `{wc26_key[0]}` 계수: **{wc26_coef_ridge:+.1f}** (무규제 대비 안정화 정도 확인)\n')

lines.append('\n### 표준화 회귀계수 (Ridge)\n| 변수 | 계수 |\n|---|---|\n')
for name, coef in sorted(zip(all_names, ridge.coef_), key=lambda x: -abs(x[1])):
    lines.append(f'| {name} | {coef:.2f} |\n')

# --- (3) 기존 결과와 비교 ---
lines.append('\n## 기존 결과와 비교\n\n')
lines.append('| 모델 | 인코딩 | RMSE | R2 |\n|---|---|---|---|\n')
lines.append(f'| 랜덤포레스트 (기존 최우수) | 서수 | 4,308.82 | 0.7556 |\n')
lines.append(f'| Ridge (기존) | 서수 | 4,357.40 | 0.7501 |\n')
lines.append(f'| 선형회귀 (원-핫, 무규제) | 원-핫 | {rmse_lr:,.2f} | {r2_lr:.4f} |\n')
lines.append(f'| **Ridge (원-핫)** | 원-핫 | **{rmse_ridge:,.2f}** | **{r2_ridge:.4f}** |\n')

beats_rf = rmse_ridge < 4308.82
lines.append(f'\n**원-핫 Ridge가 랜덤포레스트를 {"이겼다" if beats_rf else "이기지 못했다"}** '
             f'(RMSE {rmse_ridge:,.2f} vs 4,308.82).\n')

with open('outputs/model/onehot_ridge_result.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/model/onehot_ridge_result.md')
print(f'(1) 무규제 선형회귀: RMSE={rmse_lr:.2f}, R2={r2_lr:.4f}, weather_code_26 계수={wc26_coef_lr}')
print(f'(2) Ridge: RMSE={rmse_ridge:.2f}, R2={r2_ridge:.4f}, weather_code_26 계수={wc26_coef_ridge}')
print(f'랜덤포레스트(기존 최우수) RMSE=4308.82 대비: {"승" if beats_rf else "패"}')
