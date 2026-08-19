"""
3단계: 데이터 품질 점검 (train=2015, test=2016)
"""
import pandas as pd
import numpy as np

train = pd.read_csv('data/processed/day_london_2015.csv')
test = pd.read_csv('data/processed/day_london_2016.csv')

lines = ['# 런던 데이터 품질 점검 결과\n']

for name, df in [('train (2015)', train), ('test (2016)', test)]:
    lines.append(f'\n## {name}\n')
    lines.append(f'- 행 수: {len(df)}\n')
    lines.append(f'- 기간: {df["dteday"].min()} ~ {df["dteday"].max()}\n')
    lines.append(f'- 결측치 총합: {int(df.isnull().sum().sum())}\n')
    lines.append(f'- 24시간 미만 관측일 수: {int((df["n_hours_observed"] < 24).sum())}\n')

    # IQR 기준 이상치 (수치형 컬럼)
    lines.append('\n### IQR 기준 이상치\n\n| 컬럼 | 이상치 건수 |\n|---|---|\n')
    for col in ['t1', 't2', 'hum', 'wind_speed', 'cnt']:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        lines.append(f'| {col} | {n_out} |\n')

lines.append('\n## 참고 사항\n')
lines.append('- `2016-09-02`는 원본(hour) 데이터에 아예 존재하지 않아 test(2016)에도 행이 없다 '
              '(365행 = 366일(윤년) - 1일). 예측·평가 대상에서 자동으로 제외된다.\n')
lines.append('- `yr`, `t1`, `is_weekend`는 ANALYSIS_PLAN.md 1절 결정에 따라 모델 입력에서 제외했다.\n')

with open('outputs/quality/data_quality_report.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/quality/data_quality_report.md')
print(f'train: {len(train)}행, test: {len(test)}행')
print(f'train 24h 미만: {(train["n_hours_observed"]<24).sum()}일, test 24h 미만: {(test["n_hours_observed"]<24).sum()}일')
