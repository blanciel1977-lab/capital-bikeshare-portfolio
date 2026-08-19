"""
런던 시간별(london_merged.csv) -> 일별 변환

규칙 출처: 02_london/docs/DAILY_CONVERSION_PLAN.md, PREPROCESSING_NOTES.md
- t1/t2/hum/wind_speed: 시간별 평균 (소수 2자리)
- cnt: 시간별 합계
- season/is_holiday/is_weekend: 하루 중 고정값 그대로 (원본 인코딩 유지, 리매핑 없음)
- weather_code: 최악 코드(값이 큰 코드)가 4시간 이상이면 그 값, 아니면 최빈값
  (최빈값 동률 시 더 나쁜 코드 우선)
- workingday: 주말도 아니고(is_weekend==0) 휴일도 아니면(is_holiday==0) 1, 아니면 0
- yr/mnth/weekday: dteday에서 파생 (weekday는 워싱턴과 동일하게 0=일요일~6=토요일)
- n_hours_observed: 그 날 실제 관측된 시간 개수
"""
import pandas as pd

RAW_PATH = 'data/preprocess/london_merged.csv'
OUT_PATH = 'data/preprocess/day_london.csv'

df = pd.read_csv(RAW_PATH, parse_dates=['timestamp'])
df['dteday'] = df['timestamp'].dt.date


def weathersit_of_day(codes: pd.Series):
    counts = codes.value_counts()
    worst_code = codes.max()
    if counts.get(worst_code, 0) >= 4:
        return worst_code
    top = counts.max()
    candidates = counts[counts == top].index
    return max(candidates)  # 동률이면 더 나쁜(큰) 코드 우선


daily = df.groupby('dteday').agg(
    t1=('t1', lambda s: round(s.mean(), 2)),
    t2=('t2', lambda s: round(s.mean(), 2)),
    hum=('hum', lambda s: round(s.mean(), 2)),
    wind_speed=('wind_speed', lambda s: round(s.mean(), 2)),
    cnt=('cnt', 'sum'),
    season=('season', 'first'),
    is_holiday=('is_holiday', 'first'),
    is_weekend=('is_weekend', 'first'),
    weather_code=('weather_code', weathersit_of_day),
    n_hours_observed=('timestamp', 'nunique'),
).reset_index()

daily['dteday'] = pd.to_datetime(daily['dteday'])
daily['yr'] = daily['dteday'].dt.year
daily['mnth'] = daily['dteday'].dt.month
daily['weekday'] = (daily['dteday'].dt.dayofweek + 1) % 7  # 0=일요일 ~ 6=토요일
daily['workingday'] = ((daily['is_weekend'] == 0) & (daily['is_holiday'] == 0)).astype(int)

daily = daily.sort_values('dteday').reset_index(drop=True)
daily['dteday'] = daily['dteday'].dt.strftime('%Y-%m-%d')

cols = ['dteday', 'yr', 'mnth', 'weekday', 'season', 'is_holiday', 'is_weekend', 'workingday',
        'weather_code', 't1', 't2', 'hum', 'wind_speed', 'cnt', 'n_hours_observed']
daily = daily[cols]

daily.to_csv(OUT_PATH, index=False)

incomplete = (daily['n_hours_observed'] < 24).sum()

print('원본 시간별 행 수:', len(df))
print('변환 후 일별 행 수:', len(daily))
print('24시간이 온전하지 않은 날 수:', incomplete, '/', len(daily))
print()
print('24시간 미만 상위 5일:')
print(daily.nsmallest(5, 'n_hours_observed')[['dteday', 'n_hours_observed']].to_string(index=False))
