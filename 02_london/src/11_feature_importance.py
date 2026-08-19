"""
11단계: 최우수 모델(랜덤포레스트, 10단계 비교 결과) 기준 변수 중요도 시각화

워싱턴은 튜닝된 XGBoost가 최우수였지만, 런던은 09_model_comparison 결과
랜덤포레스트가 근소하게 더 낮은 RMSE를 기록해 최우수 모델로 채택했다
(outputs/model/model_comparison_summary.md 참고).
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib

from common import get_feature_cols, TARGET_COL

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

train_df = pd.read_csv('data/processed/day_london_2015.csv')
feature_cols = get_feature_cols()
X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]

name_map = {
    'season': 'season(계절)', 'mnth': 'mnth(월)', 'weekday': 'weekday(요일)',
    'is_holiday': 'is_holiday(공휴일)', 'workingday': 'workingday(근무일)',
    'weather_code': 'weather_code(날씨코드)', 't2': 't2(체감기온)',
    'hum': 'hum(습도)', 'wind_speed': 'wind_speed(풍속)',
}

model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
imp.to_csv('outputs/model/best_model_feature_importance.csv', header=['importance'])

fig, ax = plt.subplots(figsize=(9, 6))
labels = [name_map.get(i, i) for i in imp.index]
bars = ax.barh(labels, imp.values, color='#4c72b0')
for b, v in zip(bars, imp.values):
    ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f'{v:.3f}', va='center', fontsize=9)
ax.set_xlabel('중요도 (feature_importances_)')
ax.set_title('랜덤포레스트(최우수 모델) 변수 중요도 - 일별 대여건수(cnt) 예측')
ax.set_xlim(0, imp.max() * 1.15)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('outputs/model/best_model_feature_importance.png', dpi=150)

print('저장 완료: outputs/model/best_model_feature_importance.png / .csv')
print(imp.sort_values(ascending=False))
