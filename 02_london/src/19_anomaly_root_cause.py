"""
19단계: 이상치 37건의 1차 원인 변수 분해 (워싱턴 z-score 방식과 동일)

연속형 변수(t2, hum, wind_speed, cnt) 각각에 대해 전체 727일 평균·표준편차 기준
|z-score|를 계산하고, 이상치 날짜별로 가장 큰 |z-score|를 가진 변수를 "1차 원인"으로 삼는다.
weather_code는 범주형이라 z-score 비교에서 제외한다(워싱턴과 동일 원칙).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

train = pd.read_csv('data/processed/day_london_2015.csv')
test = pd.read_csv('data/processed/day_london_2016.csv')
all_days = pd.concat([train, test], ignore_index=True)

anomalies = pd.read_csv('outputs/anomaly/isoforest_anomalies.csv')

cont_feats = ['t2', 'hum', 'wind_speed', 'cnt']
means = all_days[cont_feats].mean()
stds = all_days[cont_feats].std()

name_map = {'t2': '체감기온(t2)', 'hum': '습도(hum)', 'wind_speed': '풍속(wind_speed)', 'cnt': '대여수(cnt)'}
color_map = {'t2': '#2a78d6', 'hum': '#eb6834', 'wind_speed': '#1baf7a', 'cnt': '#4a3aa7'}

rows = []
for _, r in anomalies.iterrows():
    z = {f: abs((r[f] - means[f]) / stds[f]) for f in cont_feats}
    top_var = max(z, key=z.get)
    rows.append({'dteday': r['dteday'], 'top_var': top_var, 'top_z': z[top_var]})

cause_df = pd.DataFrame(rows)
freq = cause_df['top_var'].value_counts().reindex(cont_feats, fill_value=0)

# --- 1) 원인 변수 빈도 막대 그래프 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
labels = [name_map[c] for c in cont_feats]
colors = [color_map[c] for c in cont_feats]
bars = ax.barh(labels, freq.values, color=colors)
for b, v in zip(bars, freq.values):
    pct = v / len(cause_df) * 100
    ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f'{v}건 ({pct:.0f}%)', va='center', fontsize=10)
ax.set_xlabel('1차 원인으로 나타난 이상치 건수')
ax.set_title(f'이상치 {len(cause_df)}건의 1차 원인 변수 빈도')
ax.set_xlim(0, freq.max() * 1.3)
ax.grid(axis='x', alpha=0.3)

# --- 2) t2 x hum 산점도, 색으로 원인 변수 구분 ---
ax2 = axes[1]
all_days_plot = all_days.merge(cause_df[['dteday', 'top_var']], on='dteday', how='left')
normal = all_days_plot[all_days_plot['top_var'].isna()]
ax2.scatter(normal['t2'], normal['hum'], s=12, color='#dcdcdc', alpha=0.6, label=f'정상 ({len(normal)}일)')
for c in cont_feats:
    sub = all_days_plot[all_days_plot['top_var'] == c]
    if len(sub) == 0:
        continue
    ax2.scatter(sub['t2'], sub['hum'], s=40, color=color_map[c], alpha=0.9,
                edgecolors='white', linewidths=0.5, label=f'{name_map[c]} 원인 ({len(sub)}건)')
ax2.set_xlabel('t2 (체감기온, C)')
ax2.set_ylabel('hum (습도, %)')
ax2.set_title('이상치의 위치와 1차 원인 변수')
ax2.legend(loc='lower left', fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/anomaly/isoforest_root_cause.png', dpi=150)

cause_df.merge(anomalies[['dteday', 'weather_code', 'cnt']], on='dteday').to_csv(
    'outputs/anomaly/isoforest_root_cause.csv', index=False)

print('저장 완료: outputs/anomaly/isoforest_root_cause.png, isoforest_root_cause.csv')
print(freq)
