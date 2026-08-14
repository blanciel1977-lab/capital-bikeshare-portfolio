"""
9단계: 튜닝된 XGBoost 기준 변수 중요도 시각화
"""
import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib

from common import load_best_params

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

train_df = pd.read_csv('data/train.csv')
feature_cols = [c for c in train_df.columns if c not in ['instant', 'dteday', 'cnt', 'casual', 'registered']]
X_train, y_train = train_df[feature_cols], train_df['cnt']

name_map = {
    'season': 'season(계절)', 'yr': 'yr(연도)', 'mnth': 'mnth(월)', 'holiday': 'holiday(공휴일)',
    'weekday': 'weekday(요일)', 'workingday': 'workingday(근무일)', 'weathersit': 'weathersit(날씨등급)',
    'temp': 'temp(기온)', 'atemp': 'atemp(체감온도)', 'hum': 'hum(습도)', 'windspeed': 'windspeed(풍속)',
}

model = XGBRegressor(**load_best_params())   # 07단계 튜닝 결과를 그대로 사용
model.fit(X_train, y_train)

imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
imp.to_csv('outputs/model/xgboost_feature_importance.csv', header=['importance'])

fig, ax = plt.subplots(figsize=(9, 6))
labels = [name_map.get(i, i) for i in imp.index]
bars = ax.barh(labels, imp.values, color='#4c72b0')
for b, v in zip(bars, imp.values):
    ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f'{v:.3f}', va='center', fontsize=9)
ax.set_xlabel('중요도 (feature_importances_)')
ax.set_title('XGBoost(튜닝 후) 변수 중요도 - 총 대여건수(cnt) 예측')
ax.set_xlim(0, imp.max() * 1.15)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('outputs/model/xgboost_feature_importance.png', dpi=150)

print('저장 완료: outputs/model/xgboost_feature_importance.png / .csv')
print(imp.sort_values(ascending=False))
