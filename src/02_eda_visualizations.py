"""
2단계: 탐색적 데이터 분석(EDA) 시각화
- 기온 vs 총 대여량 산점도 / hexbin 밀도 플롯
- 날씨 등급별 대여량 박스플롯
- 계절 x 요일 평균 대여량 히트맵
결과를 outputs/eda/에 저장한다.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('data/day.csv')

# --- 1. 기온 vs 대여량 산점도 ---
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(df['temp'], df['cnt'], c=df['season'], cmap='viridis', alpha=0.7, s=25)
ax.set_xlabel('정규화된 기온 (temp)')
ax.set_ylabel('총 대여 건수 (cnt)')
ax.set_title('기온과 자전거 대여 건수의 관계')
legend1 = ax.legend(*scatter.legend_elements(), title='계절\n1봄 2여름 3가을 4겨울', loc='upper left')
ax.add_artist(legend1)
corr = df['temp'].corr(df['cnt'])
ax.text(0.98, 0.05, f'상관계수(r) = {corr:.3f}', transform=ax.transAxes, fontsize=11,
        ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
plt.tight_layout()
plt.savefig('outputs/eda/temp_vs_cnt_scatter.png', dpi=150)
plt.close(fig)

# --- 2. 기온 vs 대여량 hexbin ---
fig, ax = plt.subplots(figsize=(9, 6))
hb = ax.hexbin(df['temp'], df['cnt'], gridsize=25, cmap='viridis', mincnt=1)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('밀집도(해당 구간 일수)')
ax.set_xlabel('정규화된 기온 (temp)')
ax.set_ylabel('총 대여 건수 (cnt)')
ax.set_title('기온과 자전거 대여 건수의 관계 (Hexbin 밀도 플롯)')
ax.text(0.98, 0.05, f'상관계수(r) = {corr:.3f}', transform=ax.transAxes, fontsize=11,
        ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
plt.tight_layout()
plt.savefig('outputs/eda/temp_vs_cnt_hexbin.png', dpi=150)
plt.close(fig)

# --- 3. 날씨 등급별 박스플롯 ---
weather_map = {1: '맑음', 2: '안개/흐림', 3: '약한눈/비'}
df['weather_name'] = df['weathersit'].map(weather_map)
order = ['맑음', '안개/흐림', '약한눈/비']
fig, ax = plt.subplots(figsize=(8, 6))
data = [df[df['weather_name'] == w]['cnt'].values for w in order]
counts = [len(d) for d in data]
bp = ax.boxplot(data, labels=[f'{w}\n(n={n})' for w, n in zip(order, counts)], patch_artist=True)
colors = ['#f2c14e', '#7fa6c7', '#5b6c8f']
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)
ax.set_ylabel('총 대여 건수 (cnt)')
ax.set_xlabel('날씨 등급')
ax.set_title('날씨 등급별 자전거 대여 건수 분포')
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('outputs/eda/weathersit_vs_cnt_boxplot.png', dpi=150)
plt.close(fig)

# --- 4. 계절 x 요일 히트맵 ---
season_map = {1: '봄', 2: '여름', 3: '가을', 4: '겨울'}
weekday_map = {0: '일', 1: '월', 2: '화', 3: '수', 4: '목', 5: '금', 6: '토'}
df['season_name'] = df['season'].map(season_map)
df['weekday_name'] = df['weekday'].map(weekday_map)
season_order = ['봄', '여름', '가을', '겨울']
weekday_order = ['일', '월', '화', '수', '목', '금', '토']
pivot = df.pivot_table(index='season_name', columns='weekday_name', values='cnt', aggfunc='mean')
pivot = pivot.reindex(index=season_order, columns=weekday_order)

fig, ax = plt.subplots(figsize=(9, 5.5))
im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(weekday_order)))
ax.set_xticklabels(weekday_order)
ax.set_yticks(range(len(season_order)))
ax.set_yticklabels(season_order)
for i in range(len(season_order)):
    for j in range(len(weekday_order)):
        val = pivot.values[i, j]
        color = 'white' if val > pivot.values.max() * 0.6 else 'black'
        ax.text(j, i, f'{val:,.0f}', ha='center', va='center', color=color, fontsize=9)
cb = fig.colorbar(im, ax=ax)
cb.set_label('평균 대여 건수 (cnt)')
ax.set_title('계절 x 요일별 평균 자전거 대여 건수')
ax.set_xlabel('요일')
ax.set_ylabel('계절')
plt.tight_layout()
plt.savefig('outputs/eda/season_weekday_heatmap.png', dpi=150)
plt.close(fig)

print('저장 완료: outputs/eda/ 4개 시각화')
