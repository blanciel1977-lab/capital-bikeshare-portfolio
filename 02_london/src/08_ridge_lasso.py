"""
8단계: Ridge / Lasso 회귀 (train=2015 -> test=2016)

alpha는 RidgeCV/LassoCV + TimeSeriesSplit(train 2015 내부에서만)으로 탐색한다.
XGBoost 튜닝(9단계)과 같은 원칙 — 무작위 CV는 시계열에서 낙관적인 점수를 주므로 쓰지 않는다.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from common import get_feature_cols, TARGET_COL

train_df = pd.read_csv('data/processed/day_london_2015.csv')
test_df = pd.read_csv('data/processed/day_london_2016.csv')

feature_cols = get_feature_cols()
X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

# 선형모델의 정규화(alpha)는 변수 스케일에 민감하므로 표준화 후 학습
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

cv = TimeSeriesSplit(n_splits=5)
alphas = np.logspace(-2, 3, 60)

lines = ['# Ridge / Lasso 회귀 결과 (train=2015, test=2016)\n']
lines.append('alpha 탐색: RidgeCV/LassoCV + 5-fold TimeSeriesSplit(train 2015 내부에서만), 표준화된 입력 사용\n')

results = {}
for name, Model in [('Ridge', RidgeCV), ('Lasso', LassoCV)]:
    kwargs = {'alphas': alphas, 'cv': cv}
    if name == 'Lasso':
        kwargs = {'alphas': alphas, 'cv': cv, 'max_iter': 20000}
    model = Model(**kwargs)
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results[name] = (rmse, mae, r2)

    n_zero = int((np.abs(model.coef_) < 1e-8).sum()) if name == 'Lasso' else 0

    lines.append(f'\n## {name}\n')
    lines.append(f'- 선택된 alpha: {model.alpha_:.4f}\n')
    lines.append(f'- RMSE: {rmse:.2f}\n- MAE: {mae:.2f}\n- R2: {r2:.4f}\n')
    if name == 'Lasso':
        lines.append(f'- 계수가 0으로 수렴한 변수 수: {n_zero} / {len(feature_cols)}\n')
    lines.append('\n### 표준화 회귀계수\n| 변수 | 계수 |\n|---|---|\n')
    for c, coef in sorted(zip(feature_cols, model.coef_), key=lambda x: -abs(x[1])):
        flag = ' (0으로 수렴)' if name == 'Lasso' and abs(coef) < 1e-8 else ''
        lines.append(f'| {c}{flag} | {coef:.2f} |\n')

with open('outputs/model/ridge_lasso_result.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/model/ridge_lasso_result.md')
for name, (rmse, mae, r2) in results.items():
    print(f'{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}')
