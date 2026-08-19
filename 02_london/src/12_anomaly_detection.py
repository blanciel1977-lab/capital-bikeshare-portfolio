"""
12단계: Isolation Forest 이상치 탐지 (2015+2016 전체, 727일)
워싱턴과 동일 설정(contamination=5%)으로 시작 (사용자 확정).
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

train = pd.read_csv('data/processed/day_london_2015.csv')
test = pd.read_csv('data/processed/day_london_2016.csv')
df = pd.concat([train, test], ignore_index=True).sort_values('dteday').reset_index(drop=True)

features = ['t2', 'hum', 'wind_speed', 'weather_code', 'cnt']
X = df[features]

iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
pred = iso.fit_predict(X)
score = iso.decision_function(X)

df['anomaly_score'] = score
df['is_anomaly'] = pred == -1

anomalies = df[df['is_anomaly']].copy().sort_values('anomaly_score')

for col in features:
    anomalies[col + '_pct'] = anomalies[col].apply(lambda v: (df[col] < v).mean() * 100)

cols = ['dteday', 'yr', 'season', 'weather_code', 't2', 'hum', 'wind_speed', 'cnt',
        'anomaly_score'] + [c + '_pct' for c in features]

anomalies[cols].to_csv('outputs/anomaly/isoforest_anomalies.csv', index=False)

with open('outputs/anomaly/isoforest_summary.md', 'w', encoding='utf-8') as f:
    f.write('# Isolation Forest 이상치 탐지 결과 (2015+2016, 727일)\n\n')
    f.write(f'- 사용 변수: {", ".join(features)}\n')
    f.write(f'- contamination: 5%\n')
    f.write(f'- 전체 {len(df)}일 중 이상치 {len(anomalies)}일 ({len(anomalies)/len(df)*100:.2f}%)\n\n')
    f.write('## 이상 정도가 가장 강한 상위 10일\n\n')
    f.write('| 날짜 | 연도 | weather_code | t2 | hum | wind_speed | cnt | anomaly_score |\n|---|---|---|---|---|---|---|---|\n')
    for _, r in anomalies.head(10).iterrows():
        f.write(f"| {r['dteday']} | {int(r['yr'])} | {int(r['weather_code'])} | {r['t2']:.1f} | "
                f"{r['hum']:.1f} | {r['wind_speed']:.1f} | {int(r['cnt'])} | {r['anomaly_score']:.4f} |\n")

print(f'전체 {len(df)}일 중 이상치 {len(anomalies)}일 ({len(anomalies)/len(df)*100:.2f}%)')
print()
print('상위 10일:')
print(anomalies[['dteday', 'weather_code', 't2', 'hum', 'wind_speed', 'cnt']].head(10).to_string(index=False))
