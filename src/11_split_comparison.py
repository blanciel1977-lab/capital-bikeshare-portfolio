"""
11단계: 분할 방식이 성능 평가에 미치는 영향 비교

동일한 데이터·동일한 하이퍼파라미터로 '시험 보는 방식'만 바꿔가며 측정한다.

A) 무작위 분할   : 날짜를 섞어서 20%를 테스트로 사용
                   -> 테스트 날짜의 앞뒤 날이 학습셋에 있어 성능이 과대평가됨
B) 시간순 분할   : 앞 80% 학습, 뒤 20% 예측 (본 프로젝트가 채택한 방식)
C) 연도 간 분할  : 2011년으로 학습해 2012년 예측 (미래 일반화의 극단 사례)

C의 R2가 음수(= 평균만 예측하는 것보다 나쁨)로 나오는 것이,
`yr` 변수가 미래로 외삽되지 않는다는 문제(10단계 detrend 분석의 출발점)의
직접적인 근거가 된다.

결과를 outputs/model/split_comparison_result.md에 저장한다.
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from common import load_best_params

df = pd.read_csv('data/day.csv').sort_values('instant').reset_index(drop=True)
feature_cols = [c for c in df.columns if c not in ['instant', 'dteday', 'cnt', 'casual', 'registered']]

# 세 방식 모두 동일 하이퍼파라미터(07단계 튜닝 결과)로 고정해, 분할 방식의 효과만 비교한다.
PARAMS = load_best_params()


def evaluate(train_df, test_df):
    model = XGBRegressor(**PARAMS)
    model.fit(train_df[feature_cols], train_df['cnt'])
    pred = model.predict(test_df[feature_cols])
    return (np.sqrt(mean_squared_error(test_df['cnt'], pred)),
            mean_absolute_error(test_df['cnt'], pred),
            r2_score(test_df['cnt'], pred))


results = []

# A) 무작위 분할
tr_a, te_a = train_test_split(df, test_size=0.2, random_state=42)
results.append(('A) 무작위 분할 (날짜를 섞음)', *evaluate(tr_a, te_a)))

# B) 시간순 분할
n = int(len(df) * 0.8)
results.append(('B) 시간순 분할 (앞 80% -> 뒤 20%)', *evaluate(df.iloc[:n], df.iloc[n:])))

# C) 연도 간 분할
results.append(('C) 2011년 학습 -> 2012년 예측', *evaluate(df[df['yr'] == 0], df[df['yr'] == 1])))

# 참고 baseline: 학습셋 평균만으로 예측 (B 기준)
baseline_pred = np.full(len(df.iloc[n:]), df.iloc[:n]['cnt'].mean())
base_rmse = np.sqrt(mean_squared_error(df.iloc[n:]['cnt'], baseline_pred))

# C 기준 baseline (2012년을 2011년 평균으로 예측)
base_c_pred = np.full((df['yr'] == 1).sum(), df[df['yr'] == 0]['cnt'].mean())
base_c_rmse = np.sqrt(mean_squared_error(df[df['yr'] == 1]['cnt'], base_c_pred))

lines = ['# 분할 방식에 따른 성능 비교\n\n']
lines.append('동일한 데이터와 하이퍼파라미터(XGBoost 튜닝값)로 **분할 방식만** 바꿔 측정했다.\n\n')
lines.append('| 분할 방식 | RMSE | MAE | R2 |\n|---|---|---|---|\n')
for name, rmse, mae, r2 in results:
    lines.append(f'| {name} | {rmse:.2f} | {mae:.2f} | {r2:.4f} |\n')

lines.append('\n## 해석\n\n')
lines.append(f'- **A vs B**: 무작위로 날짜를 섞으면 테스트 날짜의 바로 앞뒤 날이 학습셋에 포함된다. '
             f'일별 대여량과 날씨는 하루 사이에 거의 변하지 않으므로, 모델은 "기온·계절로 수요를 추론"하는 대신 '
             f'"양옆 값을 보고 가운데를 메우는" 쉬운 문제를 푼다. '
             f'실제 오차는 {results[1][1]:.0f}건인데 무작위 분할은 {results[0][1]:.0f}건으로 보고하므로, '
             f'오차를 {(1 - results[0][1] / results[1][1]) * 100:.0f}% 낮게(= 실력보다 후하게) 측정하는 셈이다.\n')
lines.append(f'- **B**: 본 프로젝트가 채택한 방식. 같은 기간 평균만 예측하는 baseline(RMSE {base_rmse:.0f})보다 '
             f'{(1 - results[1][1] / base_rmse) * 100:.0f}% 개선되어, 모델이 실제로 기여하고 있음을 확인할 수 있다.\n')
lines.append(f'- **C**: R2 {results[2][3]:.4f}(음수)는 "2012년의 실제 평균({df[df["yr"] == 1]["cnt"].mean():.0f}건)을 '
             f'미리 알고 일괄 대입하는 것보다도 못하다"는 뜻이다. R2는 테스트셋 자신의 평균을 기준으로 계산되기 때문이다.\n')
lines.append(f'  - 현실적인 대안인 "2011년 평균({df[df["yr"] == 0]["cnt"].mean():.0f}건)으로 2012년을 예측"과 비교하면 '
             f'모델(RMSE {results[2][1]:.0f})이 baseline(RMSE {base_c_rmse:.0f})보다 낫긴 하다. '
             f'그러나 두 방식 모두 오차가 2,000건대로, 2012년 일평균 대비 40% 수준이라 실용성이 없다.\n')
lines.append('  - 원인은 `yr` 변수다. 2011년만 학습한 모델에게 `yr=1`(2012년)은 한 번도 본 적 없는 값이라, '
             '트리 모델은 이를 외삽하지 못하고 자신이 아는 2011년 수준에 머문 예측을 내놓는다.\n')
lines.append('  - 이것이 `yr`을 제거하고 상대지수로 재학습하는 10단계 detrend 분석의 직접적인 근거다.\n')

with open('outputs/model/split_comparison_result.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('저장 완료: outputs/model/split_comparison_result.md')
for name, rmse, mae, r2 in results:
    print(f'{name:34s} RMSE={rmse:8.2f}  MAE={mae:7.2f}  R2={r2:.4f}')
print(f'{"(참고) B기준 평균-예측 baseline":34s} RMSE={base_rmse:8.2f}')
