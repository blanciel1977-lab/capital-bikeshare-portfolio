"""
1단계: 데이터 품질 점검
- 행/열 구조, 컬럼별 dtype, 결측치 확인
- IQR 기준 이상치 탐지
- 상/하위 5개 레코드 확인
결과를 outputs/quality/에 저장한다.
"""
import pandas as pd

df = pd.read_csv('data/day.csv')

lines = []
lines.append('# 데이터 품질 점검 결과\n')
lines.append(f'- 행/열: {df.shape[0]}행 x {df.shape[1]}열\n')

lines.append('\n## 컬럼별 dtype / 결측치\n')
lines.append('| 컬럼 | dtype | 결측치 |\n|---|---|---|\n')
for col in df.columns:
    lines.append(f'| {col} | {df[col].dtype} | {df[col].isna().sum()} |\n')
lines.append(f'\n전체 결측치 합계: {df.isna().sum().sum()}건\n')

lines.append('\n## IQR 기준 이상치 (1.5*IQR)\n')
num_cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
lines.append('| 컬럼 | 하한 | 상한 | 이상치 건수 |\n|---|---|---|---|\n')
for col in num_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    lines.append(f'| {col} | {lower:.4f} | {upper:.4f} | {n_out} |\n')

lines.append('\n## cnt 상위 5개\n')
top5 = df.nlargest(5, 'cnt')[['dteday', 'season', 'weathersit', 'temp', 'cnt']]
lines.append(top5.to_markdown(index=False) + '\n')

lines.append('\n## cnt 하위 5개\n')
bottom5 = df.nsmallest(5, 'cnt')[['dteday', 'season', 'weathersit', 'temp', 'cnt']]
lines.append(bottom5.to_markdown(index=False) + '\n')

with open('outputs/quality/data_quality_report.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/quality/data_quality_report.md')
print(f'행/열: {df.shape}, 결측치 합계: {df.isna().sum().sum()}')
