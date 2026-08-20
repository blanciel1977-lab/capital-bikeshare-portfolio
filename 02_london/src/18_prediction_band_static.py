"""
18단계: 예측 밴드(14단계 결과) 정적 시각화 (README 임베드용)

prediction_band_2016.json(트리 10~90 백분위 밴드 + 실제값)을 정적 PNG로 그린다.
인터랙티브 버전은 outputs/model/prediction_band_2016.html 참고.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

with open('outputs/model/prediction_band_2016.json', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data['days'])
df['date'] = pd.to_datetime(df['date'])
coverage = data['coverage_pct']

fig, ax = plt.subplots(figsize=(13, 6))

ax.fill_between(df['date'], df['low'], df['high'], color='#2a78d6', alpha=0.16,
                 label='예측 밴드 (트리 10~90 백분위)')
ax.plot(df['date'], df['low'], color='#2a78d6', linewidth=0.8, alpha=0.5)
ax.plot(df['date'], df['high'], color='#2a78d6', linewidth=0.8, alpha=0.5)

inside = df[df['inside']]
outside = df[~df['inside']]
ax.scatter(inside['date'], inside['actual'], s=10, color='#b9b8b2', alpha=0.7, label='실제값 - 밴드 안')
ax.scatter(outside['date'], outside['actual'], s=16, color='#e07b39', alpha=0.95, label='실제값 - 밴드 밖')

ax.set_title(f'2016년 일별 대여량: 예측 밴드 vs 실제값 (밴드 안 {inside.shape[0]}/{len(df)}일, {coverage:.1f}%)')
ax.set_xlabel('날짜')
ax.set_ylabel('일별 대여건수 (cnt)')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/model/prediction_band_2016.png', dpi=150)

print('저장 완료: outputs/model/prediction_band_2016.png')
print(f'커버리지: {coverage:.1f}%')
