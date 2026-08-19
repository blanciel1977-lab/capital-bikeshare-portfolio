"""
13단계: 최우수 모델(랜덤포레스트)의 2016년 일별 예측 밴드 + 실제값 시각화용 데이터 생성

밴드는 300개 트리 각각의 개별 예측값 분포에서 10~90 백분위수(80% 구간)로 계산한다.
이는 test(2016) 정답을 전혀 들여다보지 않고 모델 자체의 앙상블 불확실성만으로 구한 밴드다.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from common import get_feature_cols, TARGET_COL

train_df = pd.read_csv('data/processed/day_london_2015.csv')
test_df = pd.read_csv('data/processed/day_london_2016.csv')

feature_cols = get_feature_cols()
X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

X_test_arr = X_test.to_numpy()
tree_preds = np.stack([t.predict(X_test_arr) for t in model.estimators_], axis=1)  # (n_days, n_trees)
p10 = np.percentile(tree_preds, 10, axis=1)
p90 = np.percentile(tree_preds, 90, axis=1)
pred_mean = model.predict(X_test)

actual = y_test.values
inside = (actual >= p10) & (actual <= p90)
coverage = inside.mean() * 100

out = []
for i, row in test_df.iterrows():
    out.append({
        'date': row['dteday'],
        'actual': int(row[TARGET_COL]),
        'pred': float(pred_mean[i]),
        'low': float(p10[i]),
        'high': float(p90[i]),
        'inside': bool(inside[i]),
    })

with open('outputs/model/prediction_band_2016.json', 'w', encoding='utf-8') as f:
    json.dump({'coverage_pct': coverage, 'days': out}, f, ensure_ascii=False)

print(f'밴드(트리 10~90 백분위) 안에 실제값이 들어간 날: {inside.sum()} / {len(actual)} ({coverage:.1f}%)')
print(f'밴드 밖(과대/과소예측): {(~inside).sum()}일')
