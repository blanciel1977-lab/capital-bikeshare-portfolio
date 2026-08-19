"""
12단계: SHAP 분석 (최우수 모델 = 랜덤포레스트 기준)
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from common import get_feature_cols, TARGET_COL

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

train_df = pd.read_csv('data/processed/day_london_2015.csv')
test_df = pd.read_csv('data/processed/day_london_2016.csv')

feature_cols = get_feature_cols()
X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
X_test = test_df[feature_cols]

model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

name_map = {
    'season': 'season(계절)', 'mnth': 'mnth(월)', 'weekday': 'weekday(요일)',
    'is_holiday': 'is_holiday(공휴일)', 'workingday': 'workingday(근무일)',
    'weather_code': 'weather_code(날씨코드)', 't2': 't2(체감기온)',
    'hum': 'hum(습도)', 'wind_speed': 'wind_speed(풍속)',
}
display_names = [name_map.get(c, c) for c in feature_cols]

plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=display_names, show=False, plot_size=(10, 6))
plt.title('SHAP 요약 - 각 변수가 예측(2016)에 미치는 영향의 크기와 방향', fontsize=12)
plt.xlabel('SHAP 값 (해당 예측을 평균에서 얼마나 밀어올렸/내렸는지, 단위: 대여 건수)')
plt.tight_layout()
plt.savefig('outputs/model/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(9, 6))
shap.dependence_plot('t2', shap_values, X_test, feature_names=feature_cols,
                     interaction_index=None, show=False)
plt.title('체감기온(t2)에 따른 SHAP 값 변화')
plt.xlabel('t2 (체감기온, C)')
plt.ylabel('SHAP 값 (대여 건수 기여분)')
plt.tight_layout()
plt.savefig('outputs/model/shap_dependence_t2.png', dpi=150, bbox_inches='tight')
plt.close()

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_imp = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)
gain_imp = pd.Series(model.feature_importances_, index=feature_cols)

comparison = pd.DataFrame({
    'shap_mean_abs': shap_imp,
    'shap_rank': shap_imp.rank(ascending=False).astype(int),
    'importance': gain_imp[shap_imp.index],
    'importance_rank': gain_imp.rank(ascending=False).astype(int)[shap_imp.index],
})
comparison.to_csv('outputs/model/shap_importance.csv')

lines = ['# SHAP 분석 결과 (최우수 모델: 랜덤포레스트)\n\n']
lines.append('랜덤포레스트 모델의 test(2016) 예측을 변수별 기여도로 분해했다.\n\n')

lines.append('## 변수 중요도: SHAP vs feature_importances_\n\n')
lines.append('| 변수 | 평균 절대 SHAP (건) | SHAP 순위 | importance | importance 순위 |\n|---|---|---|---|---|\n')
for c in shap_imp.index:
    r = comparison.loc[c]
    lines.append(f"| {name_map.get(c, c)} | {r['shap_mean_abs']:.1f} | {int(r['shap_rank'])} | "
                 f"{r['importance']:.4f} | {int(r['importance_rank'])} |\n")

rank_diff = (comparison['shap_rank'] - comparison['importance_rank']).abs()
lines.append(f'\n두 방식의 순위 차이는 평균 {rank_diff.mean():.1f}계단, 최대 {rank_diff.max()}계단이다.\n')

lines.append('\n## 변수값과 기여도의 방향\n\n')
lines.append('| 변수 | 상관(변수값 vs SHAP) | 해석 |\n|---|---|---|\n')
for c in shap_imp.index:
    col = X_test[c]
    if col.nunique() < 2:
        lines.append(f'| {name_map.get(c, c)} | — | 테스트 구간에서 값이 고정({col.iloc[0]:g})되어 판정 불가 |\n')
        continue
    corr = np.corrcoef(col, shap_values[:, list(feature_cols).index(c)])[0, 1]
    direction = '값이 클수록 대여량 증가' if corr > 0 else '값이 클수록 대여량 감소'
    lines.append(f'| {name_map.get(c, c)} | {corr:+.3f} | {direction} |\n')
lines.append('\n- 상관이 0에 가까우면 단조 관계가 아니라 구간별로 방향이 바뀐다는 뜻이므로, 의존성 플롯을 함께 봐야 한다.\n')

with open('outputs/model/shap_analysis_result.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/model/shap_beeswarm.png, shap_dependence_t2.png,')
print('           shap_importance.csv, shap_analysis_result.md')
print(shap_imp.round(1))
