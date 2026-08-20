"""
17단계: Isolation Forest 이상치 탐지 결과 정적 시각화 (README 임베드용)

13단계에서 계산한 이상치를 t2 x hum 평면에 산점도로 그린다.
회색: 정상 727일 중 정상 판정, 색: 37건 이상치(연도별로 구분), 상위 5일은 날짜로 라벨링.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

train = pd.read_csv('data/processed/day_london_2015.csv')
test = pd.read_csv('data/processed/day_london_2016.csv')
all_days = pd.concat([train, test], ignore_index=True)

anomalies = pd.read_csv('outputs/anomaly/isoforest_anomalies.csv')
anomaly_dates = set(anomalies['dteday'])

all_days['is_anomaly'] = all_days['dteday'].isin(anomaly_dates)

top5_dates = set(anomalies.sort_values('anomaly_score').head(5)['dteday'])

fig, ax = plt.subplots(figsize=(9, 7))

normal = all_days[~all_days['is_anomaly']]
anom = all_days[all_days['is_anomaly']]

ax.scatter(normal['t2'], normal['hum'], s=14, color='#c9c9c9', alpha=0.6, label=f'정상 ({len(normal)}일)')
ax.scatter(anom['t2'], anom['hum'], s=34, color='#e07b39', alpha=0.9,
           edgecolors='white', linewidths=0.5, label=f'이상치 ({len(anom)}일, 5.09%)')

for _, row in anom[anom['dteday'].isin(top5_dates)].iterrows():
    ax.annotate(row['dteday'], (row['t2'], row['hum']),
                textcoords='offset points', xytext=(6, 6), fontsize=9, fontweight='bold')

ax.set_xlabel('t2 (체감기온, C)')
ax.set_ylabel('hum (습도, %)')
ax.set_title('Isolation Forest 이상치 탐지 결과 (2015~2016, 727일 중 37일)')
ax.legend(loc='lower left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/anomaly/isoforest_anomalies.png', dpi=150)

print('저장 완료: outputs/anomaly/isoforest_anomalies.png')
