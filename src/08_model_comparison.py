"""
8단계: 4개 모델(선형회귀/랜덤포레스트/XGBoost 기본/XGBoost 튜닝) 성능 비교 시각화
※ 이 스크립트를 실행하기 전에 04, 05, 06, 07번 스크립트를 먼저 실행해
  outputs/model/ 아래 각 결과 md 파일이 생성되어 있어야 한다.
"""
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


def parse_metrics(path):
    text = open(path, encoding='utf-8').read()
    # 여러 RMSE/R2 값이 등장할 수 있으므로(예: CV 점수), 마지막에 등장하는
    # test set 최종 성능 수치를 사용한다.
    rmse = float(re.findall(r'RMSE:\s*([\d.]+)', text)[-1])
    mae = float(re.findall(r'MAE:\s*([\d.]+)', text)[-1])
    r2 = float(re.findall(r'R2:\s*([\d.]+)', text)[-1])
    return rmse, mae, r2


# 선형회귀 결과 파일은 (1)누수 재현 + (2)정상 학습 두 블록이 있으므로 두 번째 블록만 사용
lr_text = open('outputs/model/linear_regression_result.md', encoding='utf-8').read()
lr_block = lr_text.split('## (2)')[1]
lr_rmse = float(re.search(r'RMSE:\s*([\d.]+)', lr_block).group(1))
lr_mae = float(re.search(r'MAE:\s*([\d.]+)', lr_block).group(1))
lr_r2 = float(re.search(r'R2:\s*([\d.]+)', lr_block).group(1))

rf_rmse, rf_mae, rf_r2 = parse_metrics('outputs/model/random_forest_result.md')
xgb_rmse, xgb_mae, xgb_r2 = parse_metrics('outputs/model/xgboost_baseline_result.md')
xgbt_rmse, xgbt_mae, xgbt_r2 = parse_metrics('outputs/model/xgboost_tuning_result.md')

models = ['선형회귀', '랜덤포레스트', 'XGBoost\n(기본)', 'XGBoost\n(튜닝)']
rmse = [lr_rmse, rf_rmse, xgb_rmse, xgbt_rmse]
mae = [lr_mae, rf_mae, xgb_mae, xgbt_mae]
r2 = [lr_r2, rf_r2, xgb_r2, xgbt_r2]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width / 2, rmse, width, label='RMSE', color='#4c72b0')
bars2 = ax.bar(x + width / 2, mae, width, label='MAE', color='#dd8452')
for bars in (bars1, bars2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 10, f'{h:,.0f}', ha='center', va='bottom', fontsize=9)
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

fig.suptitle('모델별 성능 비교', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('outputs/model/model_comparison.png', dpi=150, bbox_inches='tight')

print('저장 완료: outputs/model/model_comparison.png')
