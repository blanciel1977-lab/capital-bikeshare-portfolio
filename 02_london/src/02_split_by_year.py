"""
일별 데이터(day_london.csv)를 연도별로 분리

2015/2016년만 각각 저장하고, 2017년(1~3일치, 불완전한 연도)은 제외한다.
"""
import pandas as pd

IN_PATH = 'data/preprocess/day_london.csv'
OUT_2015 = 'data/processed/day_london_2015.csv'
OUT_2016 = 'data/processed/day_london_2016.csv'

df = pd.read_csv(IN_PATH)

d2015 = df[df['yr'] == 2015]
d2016 = df[df['yr'] == 2016]
d2017 = df[df['yr'] == 2017]

d2015.to_csv(OUT_2015, index=False)
d2016.to_csv(OUT_2016, index=False)

print(f'2015: {len(d2015)}행 ({d2015["dteday"].min()} ~ {d2015["dteday"].max()}) -> {OUT_2015}')
print(f'2016: {len(d2016)}행 ({d2016["dteday"].min()} ~ {d2016["dteday"].max()}) -> {OUT_2016}')
print(f'2017: {len(d2017)}행 제외됨')
