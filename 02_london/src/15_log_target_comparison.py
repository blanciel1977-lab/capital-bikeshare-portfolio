"""
15단계: log(cnt) 변환 후 6개 모델 재비교 (보조 실험, 09단계 원 스케일 비교와 대조용)

cnt는 오른쪽으로 치우친(right-skewed) 분포라, 많은 공개 분석이 log1p(cnt)로 변환한 뒤
모델링한다. 로그 변환이 "최우수 모델" 순위를 실제로 바꾸는지 확인한다.

- 학습은 log1p(cnt) 기준으로 하되, 평가는 expm1로 되돌린 원 스케일 RMSE/MAE/R2로 한다
  (09단계 결과와 직접 비교 가능하도록).
- 하이퍼파라미터는 원 스케일 비교(05~09단계)에서 쓴 것과 동일하게 고정한다
  (튜닝 자체를 로그 타깃으로 다시 하면 두 실험이 서로 다른 것을 비교하게 되므로).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from common import get_feature_cols, TARGET_COL, load_best_params

train_df = pd.read_csv('data/processed/day_london_2015.csv')
test_df = pd.read_csv('data/processed/day_london_2016.csv')

feature_cols = get_feature_cols()
X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

y_train_log = np.log1p(y_train)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

cv = TimeSeriesSplit(n_splits=5)
alphas = np.logspace(-2, 3, 60)

results = {}


def evaluate(name, pred_log):
    pred = np.expm1(pred_log)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results[name] = (rmse, mae, r2)
    print(f'{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}')


# 1) 선형회귀
lr = LinearRegression()
lr.fit(X_train, y_train_log)
evaluate('선형회귀', lr.predict(X_test))

# 2) Ridge
ridge = RidgeCV(alphas=alphas, cv=cv)
ridge.fit(X_train_s, y_train_log)
evaluate('Ridge', ridge.predict(X_test_s))

# 3) Lasso
lasso = LassoCV(alphas=alphas, cv=cv, max_iter=20000)
lasso.fit(X_train_s, y_train_log)
evaluate('Lasso', lasso.predict(X_test_s))

# 4) 랜덤포레스트 (원 스케일 비교와 동일 설정)
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_log)
evaluate('랜덤포레스트', rf.predict(X_test))

# 5) XGBoost 기본값
xgb_base = XGBRegressor(random_state=42, n_jobs=-1)
xgb_base.fit(X_train, y_train_log)
evaluate('XGBoost(기본)', xgb_base.predict(X_test))

# 6) XGBoost 튜닝 (09단계에서 찾은 최적 하이퍼파라미터 재사용)
xgb_tuned = XGBRegressor(**load_best_params())
xgb_tuned.fit(X_train, y_train_log)
evaluate('XGBoost(튜닝)', xgb_tuned.predict(X_test))

# --- 결과 저장 ---
rows = sorted(results.items(), key=lambda x: x[1][0])
best_name, (best_rmse, best_mae, best_r2) = rows[0]

lines = ['# log(cnt) 변환 후 모델 재비교 (train=2015, test=2016)\n\n']
lines.append('학습은 log1p(cnt) 기준, 평가는 expm1로 원 스케일로 되돌린 RMSE/MAE/R2 (09단계 원 스케일 비교와 직접 비교 가능)\n\n')
lines.append('| 순위 | 모델 | RMSE | MAE | R2 |\n|---|---|---|---|---|\n')
for i, (name, (rmse, mae, r2)) in enumerate(rows, 1):
    mark = '**' if i == 1 else ''
    lines.append(f'| {i} | {mark}{name}{mark} | {mark}{rmse:,.2f}{mark} | {mark}{mae:,.2f}{mark} | {mark}{r2:.4f}{mark} |\n')

lines.append(f'\n**최우수 모델(로그 변환): {best_name}**\n')

lines.append('\n## 원 스케일 비교(09단계)와의 순위 대조\n\n')
lines.append('| 모델 | 원 스케일 RMSE | 로그변환 RMSE | 순위 변화 |\n|---|---|---|---|\n')
orig = {
    '선형회귀': 4359.46, 'Ridge': 4357.40, 'Lasso': 4351.50,
    '랜덤포레스트': 4308.82, 'XGBoost(기본)': 4931.37, 'XGBoost(튜닝)': 4419.45,
}
orig_rank = {name: i + 1 for i, name in enumerate(sorted(orig, key=orig.get))}
log_rank = {name: i + 1 for i, (name, _) in enumerate(rows)}
for name in orig:
    diff = orig_rank[name] - log_rank[name]
    arrow = f'{orig_rank[name]}위 -> {log_rank[name]}위' + (f' ({"+" if diff>0 else ""}{diff})' if diff != 0 else ' (동일)')
    lines.append(f'| {name} | {orig[name]:,.2f} | {results[name][0]:,.2f} | {arrow} |\n')

with open('outputs/model/log_target_comparison.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print()
print('저장 완료: outputs/model/log_target_comparison.md')
print(f'로그변환 최우수 모델: {best_name}, RMSE={best_rmse:.2f}, R2={best_r2:.4f}')
