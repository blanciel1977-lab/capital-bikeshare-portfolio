"""
10단계: yr(연도) 변수의 재사용성 문제 보정 - detrend 분석
- yr은 2011/2012 두 범주뿐인 이진 변수라 미래 연도로 외삽할 수 없다.
- cnt를 '그 해 평균 대비 상대지수'로 바꾸고 yr을 제외한 뒤,
  날씨/계절/요일만으로 재학습해 연도와 무관하게 재사용 가능한 요인 구조를 확인한다.
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from common import load_best_params

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# 연도별 기준선(평균)은 반드시 '학습셋만'으로 계산한다.
# 전체 데이터로 평균을 내면 테스트 구간의 정보가 타깃에 섞여 들어가(누수),
# 하필 '일반화 가능성'을 검증하는 이 분석의 결론이 낙관적으로 왜곡된다.
yr_baseline = train_df.groupby('yr')['cnt'].mean()

train_df['cnt_relative'] = train_df['cnt'] / train_df['yr'].map(yr_baseline)
test_df['cnt_relative'] = test_df['cnt'] / test_df['yr'].map(yr_baseline)

feature_cols = [c for c in train_df.columns
                 if c not in ['instant', 'dteday', 'cnt', 'casual', 'registered', 'yr', 'cnt_relative']]
X_train, y_train = train_df[feature_cols], train_df['cnt_relative']
X_test, y_test = test_df[feature_cols], test_df['cnt_relative']

model = XGBRegressor(**load_best_params())   # 07단계 튜닝 결과를 그대로 사용
model.fit(X_train, y_train)
pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

lines = ['# yr 변수 제거 + detrend 분석\n']
lines.append('## 배경\n')
lines.append('`yr`은 2011/2012 두 범주뿐인 이진 변수이며, 튜닝 모델에서 중요도 34%로 1위였다. '
              '이는 "날씨/계절의 진짜 효과"가 아니라 "그 2년간 서비스가 이만큼 성장했다"는 사실을 '
              '암기한 결과로, 미래 연도(2013년 이후)에는 재사용할 수 없다.\n')
lines.append('\n## 보정 방법\n')
lines.append('`cnt_relative = cnt / (해당 연도 평균 cnt)` 로 변환해 "그 해 평균 대비 몇 배인가"라는 '
              '상대지수를 목표변수로 삼고, `yr`을 입력에서 제외한 뒤 날씨/계절/요일 변수로 재학습한다.\n')
lines.append(f'\n## 성능 (상대지수 단위, 절대치 모델과 직접 비교 불가)\n')
lines.append(f'- RMSE: {rmse:.4f}\n- MAE: {mae:.4f}\n- R2: {r2:.4f}\n')
lines.append('\n## 변수 중요도 (yr 제외)\n| 변수 | 중요도 |\n|---|---|\n')
for c, imp in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
    lines.append(f'| {c} | {imp:.4f} |\n')
lines.append('\n## 실무 적용: 2단계 예측 파이프라인\n')
lines.append('1. 연도별 기준선(baseline) 추정 - 과거 성장 추세(예: 2011->2012 +64%) 또는 '
              '등록회원 증가율 등 외부 지표로 미래 연도의 평균 대여량 추정\n')
lines.append('2. 이 모델로 날씨/계절/요일 조건에 따른 "기준선 대비 상대지수" 예측\n')
lines.append('3. 최종 예측 = 1의 기준선 x 2의 상대지수\n')

with open('outputs/model/detrend_analysis_result.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/model/detrend_analysis_result.md')
print(f'RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}')
