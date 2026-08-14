"""
3단계: 데이터 분할 (train 80% / test 20%)
결과를 data/train.csv, data/test.csv로 저장한다.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/day.csv')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.sort_values('instant')
test_df = test_df.sort_values('instant')

train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print('전체:', len(df))
print('train:', len(train_df), f'({len(train_df) / len(df) * 100:.1f}%)')
print('test:', len(test_df), f'({len(test_df) / len(df) * 100:.1f}%)')
