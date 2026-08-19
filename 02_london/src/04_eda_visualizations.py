"""
4단계: EDA 시각화 (워싱턴 02_eda_visualizations.py와 동일 구성)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

train = pd.read_csv('data/processed/day_london_2015.csv')
train['weather_code'] = train['weather_code'].astype(int)

# 1) 체감기온(t2) vs 대여량 산점도
plt.figure(figsize=(8, 6))
plt.scatter(train['t2'], train['cnt'], alpha=0.4, s=15)
corr = train['t2'].corr(train['cnt'])
plt.title(f'체감기온(t2) vs 대여량 (train=2015, r={corr:.3f})')
plt.xlabel('t2 (체감기온, C)')
plt.ylabel('cnt (일별 대여건수)')
plt.tight_layout()
plt.savefig('outputs/eda/t2_vs_cnt_scatter.png', dpi=120)
plt.close()

# 2) 체감기온 vs 대여량 hexbin
plt.figure(figsize=(8, 6))
plt.hexbin(train['t2'], train['cnt'], gridsize=25, cmap='Blues')
plt.colorbar(label='count')
plt.title('체감기온(t2) vs 대여량 (hexbin, train=2015)')
plt.xlabel('t2 (체감기온, C)')
plt.ylabel('cnt')
plt.tight_layout()
plt.savefig('outputs/eda/t2_vs_cnt_hexbin.png', dpi=120)
plt.close()

# 3) weather_code별 대여량 boxplot (원본 코드 순서대로)
order = [c for c in [1, 2, 3, 4, 7, 10, 26] if c in set(train['weather_code'].unique())]
plt.figure(figsize=(9, 6))
sns.boxplot(data=train, x='weather_code', y='cnt', order=order)
plt.title('날씨코드(weather_code)별 대여량 분포 (train=2015)')
plt.xlabel('weather_code (1=맑음 -> 26=눈, 악화 순)')
plt.ylabel('cnt')
plt.tight_layout()
plt.savefig('outputs/eda/weather_code_vs_cnt_boxplot.png', dpi=120)
plt.close()

# 4) 계절 x 요일 히트맵
pivot = train.pivot_table(index='season', columns='weekday', values='cnt', aggfunc='mean')
plt.figure(figsize=(9, 5))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('계절(season) x 요일(weekday) 평균 대여량 (train=2015)')
plt.xlabel('weekday (0=일 ~ 6=토)')
plt.ylabel('season (0=봄,1=여름,2=가을,3=겨울)')
plt.tight_layout()
plt.savefig('outputs/eda/season_weekday_heatmap.png', dpi=120)
plt.close()

# 상관관계 요약 저장
corr_series = train.corr(numeric_only=True)['cnt'].sort_values(ascending=False)
with open('outputs/eda/correlation_summary.md', 'w', encoding='utf-8') as f:
    f.write('# cnt 기준 상관관계 (train=2015)\n\n| 변수 | 상관계수 |\n|---|---|\n')
    for k, v in corr_series.items():
        if k == 'cnt':
            continue
        f.write(f'| {k} | {v:.3f} |\n')

print('EDA 시각화 4종 + 상관관계 요약 저장 완료 (outputs/eda/)')
print(corr_series)
