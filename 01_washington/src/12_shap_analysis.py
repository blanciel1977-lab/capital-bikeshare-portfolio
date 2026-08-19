"""
12단계: SHAP 분석 — 요인의 '크기'뿐 아니라 '방향과 모양'까지 설명

gain 기반 feature_importances_(9단계)는 "이 변수가 얼마나 중요한가"라는 크기만
알려줄 뿐, "값이 커지면 수요가 늘어나는가 줄어드는가", "어느 구간부터 효과가
꺾이는가" 같은 관계의 형태는 알려주지 않는다.

SHAP은 각 예측을 변수별 기여도로 분해하므로 다음을 얻을 수 있다.
  1) beeswarm : 변수별 영향의 크기 + 방향(빨강=변수값 높음)을 한 장에
  2) 의존성 플롯 : 특정 변수의 값이 변할 때 예측이 어떻게 움직이는지(비선형 포함)

또한 gain 중요도와 나란히 비교해, 두 방식의 순위가 얼마나 일치하는지 확인한다.

결과: outputs/model/shap_beeswarm.png
      outputs/model/shap_dependence_atemp.png
      outputs/model/shap_importance.csv
      outputs/model/shap_analysis_result.md
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from common import get_feature_cols, load_best_params

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

feature_cols = get_feature_cols(train_df)
X_train, y_train = train_df[feature_cols], train_df['cnt']
X_test = test_df[feature_cols]

model = XGBRegressor(**load_best_params())
model.fit(X_train, y_train)

# SHAP 값은 test set 기준으로 계산한다(학습에 쓰지 않은 구간에서의 설명).
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

name_map = {
    'season': 'season(계절)', 'yr': 'yr(연도)', 'mnth': 'mnth(월)', 'holiday': 'holiday(공휴일)',
    'weekday': 'weekday(요일)', 'workingday': 'workingday(근무일)',
    'weathersit': 'weathersit(날씨등급)', 'atemp': 'atemp(체감온도)',
    'hum': 'hum(습도)', 'windspeed': 'windspeed(풍속)',
}
display_names = [name_map.get(c, c) for c in feature_cols]

# --- 1) beeswarm: 영향의 크기와 방향을 한 장에 ---
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=display_names, show=False, plot_size=(10, 6))
plt.title('SHAP 요약 - 각 변수가 예측에 미치는 영향의 크기와 방향', fontsize=12)
plt.xlabel('SHAP 값 (해당 예측을 평균에서 얼마나 밀어올렸/내렸는지, 단위: 대여 건수)')
plt.tight_layout()
plt.savefig('outputs/model/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

# --- 2) 의존성 플롯: 체감온도가 변할 때 예측이 어떻게 움직이는가 ---
plt.figure(figsize=(9, 6))
shap.dependence_plot('atemp', shap_values, X_test, feature_names=feature_cols,
                     interaction_index=None, show=False)
plt.title('체감온도(atemp)에 따른 SHAP 값 변화')
plt.xlabel('정규화된 체감온도 (atemp)')
plt.ylabel('SHAP 값 (대여 건수 기여분)')
plt.tight_layout()
plt.savefig('outputs/model/shap_dependence_atemp.png', dpi=150, bbox_inches='tight')
plt.close()

# --- 3) 평균 |SHAP| 기준 중요도 vs gain 중요도 비교 ---
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_imp = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)
gain_imp = pd.Series(model.feature_importances_, index=feature_cols)

comparison = pd.DataFrame({
    'shap_mean_abs': shap_imp,
    'shap_rank': shap_imp.rank(ascending=False).astype(int),
    'gain': gain_imp[shap_imp.index],
    'gain_rank': gain_imp.rank(ascending=False).astype(int)[shap_imp.index],
})
comparison.to_csv('outputs/model/shap_importance.csv')

# --- 4) 결과 md ---
lines = ['# SHAP 분석 결과\n\n']
lines.append('튜닝된 XGBoost 모델의 test set 예측을 변수별 기여도로 분해했다.\n')
lines.append('`temp`는 `atemp`와 상관 0.9917로 중복이라 입력에서 제외되어 있다(common.py 참고).\n\n')

lines.append('## 변수 중요도: SHAP vs gain\n\n')
lines.append('| 변수 | 평균 절대 SHAP (건) | SHAP 순위 | gain | gain 순위 |\n|---|---|---|---|---|\n')
for c in shap_imp.index:
    r = comparison.loc[c]
    lines.append(f"| {name_map.get(c, c)} | {r['shap_mean_abs']:.1f} | {int(r['shap_rank'])} | "
                 f"{r['gain']:.4f} | {int(r['gain_rank'])} |\n")

rank_diff = (comparison['shap_rank'] - comparison['gain_rank']).abs()
lines.append(f'\n두 방식의 순위 차이는 평균 {rank_diff.mean():.1f}계단, 최대 {rank_diff.max()}계단이다. ')
lines.append('SHAP은 "평균 예측에서 실제로 몇 건을 밀어올렸/내렸는가"를 건수 단위로 재므로, '
             '트리 내부의 분기 이득을 세는 gain보다 해석이 직관적이다.\n')

lines.append('\n## SHAP이 추가로 알려주는 것\n\n')
lines.append('gain 중요도는 크기만 알려주지만, SHAP은 **방향과 형태**를 보여준다.\n\n')
lines.append('- `outputs/model/shap_beeswarm.png`: 변수별로 점 하나가 하루를 뜻한다. '
             '점이 오른쪽에 있으면 그날 예측을 끌어올린 것이고, 색이 붉을수록 그 변수의 값이 높다. '
             '따라서 "값이 높을 때 수요가 오르는지 내리는지"를 한눈에 볼 수 있다.\n')
lines.append('- `outputs/model/shap_dependence_atemp.png`: 체감온도가 올라갈수록 기여가 어떻게 '
             '변하는지 곡선으로 보여준다. 선형회귀 계수 하나로는 표현할 수 없는 비선형 구간을 확인할 수 있다.\n')

# 방향성 요약: 변수값과 SHAP 값의 상관으로 대략적인 방향을 계산
lines.append('\n## 변수값과 기여도의 방향\n\n')
lines.append('| 변수 | 상관(변수값 vs SHAP) | 해석 |\n|---|---|---|\n')
for c in shap_imp.index[:6]:
    col = X_test[c]
    if col.nunique() < 2:
        # 테스트 구간이 2012년 하반기뿐이라 yr처럼 값이 하나로 고정된 변수가 있다.
        # 이 경우 상관을 정의할 수 없으므로 방향을 판정하지 않는다.
        lines.append(f'| {name_map.get(c, c)} | — | 테스트 구간에서 값이 고정({col.iloc[0]:g})되어 판정 불가 |\n')
        continue
    corr = np.corrcoef(col, shap_values[:, list(feature_cols).index(c)])[0, 1]
    direction = '값이 클수록 대여량 증가' if corr > 0 else '값이 클수록 대여량 감소'
    lines.append(f'| {name_map.get(c, c)} | {corr:+.3f} | {direction} |\n')
lines.append('\n- 상관이 0에 가까우면 단조 관계가 아니라 구간별로 방향이 바뀐다는 뜻이므로, '
             '위 의존성 플롯을 함께 봐야 한다.\n')
lines.append('- `yr`은 테스트 구간(2012년 하반기)에서 값이 1로 고정되어 있어 방향을 잴 수 없다. '
             '이는 5절에서 다룬 "yr은 학습 기간 안에서만 유효한 변수"라는 한계와 같은 맥락이다.\n')

with open('outputs/model/shap_analysis_result.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/model/shap_beeswarm.png, shap_dependence_atemp.png,')
print('           shap_importance.csv, shap_analysis_result.md')
print(shap_imp.round(1))
