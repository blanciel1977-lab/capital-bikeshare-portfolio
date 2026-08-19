"""
3단계: 데이터 분할 (train 80% / test 20%) — 시간순 분할

day.csv는 2011-01-01 ~ 2012-12-31의 일별 시계열이다.
날짜를 무작위로 섞어 나누면 테스트 날짜의 바로 앞뒤 날이 학습셋에 들어가는데,
일별 대여량과 날씨는 하루 사이에 거의 변하지 않으므로 모델이 '양옆 값을 보고
가운데를 메우는' 문제를 푸는 셈이 되어 성능이 과대평가된다.

실제 활용 시나리오는 "오늘까지의 데이터로 앞으로를 예측"하는 것이므로,
날짜 순으로 앞 80%를 학습, 뒤 20%를 테스트로 사용한다.
(분할 방식에 따른 성능 차이는 src/11_split_comparison.py에서 정량 비교)

결과를 data/train.csv, data/test.csv로 저장한다.
"""
import pandas as pd

df = pd.read_csv('data/day.csv').sort_values('instant').reset_index(drop=True)

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print('전체:', len(df))
print('train:', len(train_df), f'({len(train_df) / len(df) * 100:.1f}%)',
      f"{train_df['dteday'].iloc[0]} ~ {train_df['dteday'].iloc[-1]}")
print('test: ', len(test_df), f'({len(test_df) / len(df) * 100:.1f}%)',
      f"{test_df['dteday'].iloc[0]} ~ {test_df['dteday'].iloc[-1]}")
