"""
9단계: 모델 성능 비교 + baseline(train 평균 예측) 비교
"""
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from common import TARGET_COL

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


def parse_metrics(path):
    text = open(path, encoding='utf-8').read()
    rmse = float(re.findall(r'RMSE:\s*([\d.]+)', text)[-1])
    mae = float(re.findall(r'MAE:\s*([\d.]+)', text)[-1])
    r2 = float(re.findall(r'R2:\s*([\d.]+)', text)[-1])
    return rmse, mae, r2


lr_rmse, lr_mae, lr_r2 = parse_metrics('outputs/model/linear_regression_result.md')
rf_rmse, rf_mae, rf_r2 = parse_metrics('outputs/model/random_forest_result.md')
xgb_rmse, xgb_mae, xgb_r2 = parse_metrics('outputs/model/xgboost_baseline_result.md')
xgbt_rmse, xgbt_mae, xgbt_r2 = parse_metrics('outputs/model/xgboost_tuning_result.md')

# baseline: train(2015) 평균만 예측
train_df = pd.read_csv('data/processed/day_london_2015.csv')
test_df = pd.read_csv('data/processed/day_london_2016.csv')
baseline_pred = np.full(len(test_df), train_df[TARGET_COL].mean())
baseline_rmse = np.sqrt(mean_squared_error(test_df[TARGET_COL], baseline_pred))

models = ['선형회귀', '랜덤포레스트', 'XGBoost\n(기본)', 'XGBoost\n(튜닝)']
rmse = [lr_rmse, rf_rmse, xgb_rmse, xgbt_rmse]
mae = [lr_mae, rf_mae, xgb_mae, xgbt_mae]
r2 = [lr_r2, rf_r2, xgb_r2, xgbt_r2]

best_idx = int(np.argmin(rmse))
best_name = ['선형회귀', '랜덤포레스트', 'XGBoost(기본)', 'XGBoost(튜닝)'][best_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width / 2, rmse, width, label='RMSE', color='#4c72b0')
bars2 = ax.bar(x + width / 2, mae, width, label='MAE', color='#dd8452')
for bars in (bars1, bars2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 30, f'{h:,.0f}', ha='center', va='bottom', fontsize=9)
ax.axhline(baseline_rmse, color='gray', linestyle='--', linewidth=1, label=f'baseline RMSE={baseline_rmse:,.0f}')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('오차 (건수)')
ax.set_title('모델별 RMSE / MAE 비교 (낮을수록 우수)')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.4)

ax2 = axes[1]
colors = ['#c9c9c9', '#7fa6c7', '#f2c14e', '#e07b39']
bars3 = ax2.bar(models, r2, color=colors, width=0.5)
for b, v in zip(bars3, r2):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
ax2.set_ylim(0, 1)
ax2.set_ylabel('R2 (설명력)')
ax2.set_title('모델별 R2 비교 (높을수록 우수)')
ax2.grid(axis='y', linestyle='--', alpha=0.4)

fig.suptitle('모델별 성능 비교 (train=2015, test=2016)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('outputs/model/model_comparison.png', dpi=150, bbox_inches='tight')

with open('outputs/model/model_comparison_summary.md', 'w', encoding='utf-8') as f:
    f.write('# 모델 성능 비교 (train=2015, test=2016)\n\n')
    f.write('| 순위 | 모델 | RMSE | MAE | R2 |\n|---|---|---|---|---|\n')
    rows = sorted(zip(['선형회귀', '랜덤포레스트', 'XGBoost(기본)', 'XGBoost(튜닝)'], rmse, mae, r2), key=lambda x: x[1])
    for i, (name, rm, ma, r) in enumerate(rows, 1):
        mark = '**' if i == 1 else ''
        f.write(f'| {i} | {mark}{name}{mark} | {mark}{rm:,.2f}{mark} | {mark}{ma:,.2f}{mark} | {mark}{r:.4f}{mark} |\n')
    f.write(f'| — | (참고) train 평균만 예측 | {baseline_rmse:,.2f} | — | — |\n')
    f.write(f'\n**최우수 모델: {best_name}** (RMSE 기준)\n')
    f.write(f'\nbaseline 대비 개선율: {(1 - rmse[best_idx]/baseline_rmse)*100:.1f}%\n')

print('저장 완료: outputs/model/model_comparison.png, model_comparison_summary.md')
print(f'baseline(train 평균) RMSE = {baseline_rmse:,.2f}')
print(f'최우수 모델: {best_name}, RMSE={rmse[best_idx]:,.2f}, R2={r2[best_idx]:.4f}')
