# London Bike Sharing 수요 예측 (2015 학습 → 2016 예측)

런던 공유자전거(Santander Cycles) 시간별 대여 데이터를 일별로 직접 변환하는 단계부터 시작해, **2015년 데이터로 학습한 모델이 2016년 대여량을 얼마나 정확히 예측하는지** 검증한 프로젝트다. [`01_washington`](../01_washington)에서 얻은 방법론적 교훈(연도 변수의 외삽 불가, 다중공선성, 평가 설계의 함정)을 계획 단계부터 선반영했다는 점이 이 프로젝트의 핵심이다.

전체 분석 과정은 Claude Code CLI로 수행했다. 과정 기록은 [docs/CLAUDE_WORKFLOW.md](docs/CLAUDE_WORKFLOW.md), 계획 수립 과정은 [docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md)에 별도 정리했다.

## 핵심 결과 요약

- 원본: 시간별 데이터 17,414행(2015-01-04~2017-01-03) → 일별 변환 730행 → 2015/2016만 사용(2017년 3일치 제외)
- 최종 입력 변수 9개: `season, mnth, weekday, is_holiday, workingday, weather_code, t2(체감기온), hum(습도), wind_speed` — `yr`, `t1`, `is_weekend`는 실측 근거로 사전 제외(아래 참고)
- 모델 비교 결과, **랜덤포레스트가 RMSE 4,308.82 / R² 0.7556로 최우수** (train 평균만 예측하는 baseline RMSE 8,757.83 대비 50.8% 개선)
- 워싱턴은 항상 튜닝된 XGBoost가 1위였지만, 런던은 랜덤포레스트가 근소 우위 — **결과를 미리 가정하지 않고 실측으로 채택**
- 변수 중요도: 체감기온(t2) 42.1% + 습도(hum) 23.3%로 두 변수가 지배적. **런던은 습도의 영향이 워싱턴보다 훨씬 강하다**(상관 -0.587 vs 워싱턴 -0.101)
- 상세 내용은 [docs/REPORT.md](docs/REPORT.md) 참고

| 순위 | 모델 | RMSE | MAE | R2 |
|---|---|---|---|---|
| 1 | **랜덤포레스트** | **4,308.82** | **3,169.69** | **0.7556** |
| 2 | Lasso | 4,351.50 | 3,381.41 | 0.7508 |
| 3 | Ridge | 4,357.40 | 3,385.77 | 0.7501 |
| 4 | 선형회귀 | 4,359.46 | 3,387.17 | 0.7499 |
| 5 | XGBoost (튜닝) | 4,419.45 | 3,148.64 | 0.7429 |
| 6 | XGBoost (기본값) | 4,931.37 | 3,477.51 | 0.6799 |

## 워싱턴에서 배운 교훈을 처음부터 반영한 3가지

워싱턴 프로젝트는 `yr`(연도) 변수의 외삽 불가 문제, 다중공선성, 무작위 분할의 함정을 **분석을 다 끝낸 뒤에야 발견**했다. 런던에서는 같은 실수를 반복하지 않기 위해 계획 수립 단계에서 미리 실측 검증했다.

| 문제 | 확인 방법 | 조치 |
|---|---|---|
| `yr`이 train(2015)/test(2016) 각각 단일값 | 상관계수 계산 시 NaN으로 나옴을 사전 확인 | 계획 단계부터 입력에서 제외 (워싱턴은 사후 detrend로 보정) |
| `t1`(실제기온)~`t2`(체감기온) 다중공선성 | 상관계수 0.992 실측 | `t1` 제외, `t2`만 사용 |
| `is_weekend`가 `weekday`의 완전 중복 | 730일 전체에서 weekday 0·6일 때만 is_weekend=1임을 확인, 예외 없음 | `is_weekend` 제외 |

## 이상치 탐지: 날씨로 설명 안 되는 실제 사건

Isolation Forest(contamination 5%)로 2015~2016년 727일 중 37일(5.09%)을 이상치로 탐지했다. 상위권 대부분은 눈·강풍·고습도가 겹친 겨울날이었지만, **가장 눈에 띄는 발견은 2015-07-09(대여량 72,504건, 전체 최고 수준)**였다.

외부 뉴스([Cycling Weekly](https://www.cyclingweekly.com/news/latest-news/tube-strike-forces-londoners-take-bikes-305938))로 대조한 결과, 이 날은 **런던 지하철 전면파업일**로 실제 자전거 대여 서비스가 개시 이래 최다 이용일(73,094건)을 기록한 날과 정확히 일치했다. 날씨 변수만으로는 절대 설명할 수 없는 사회적 이벤트가 이상치로 정확히 잡힌 사례다 — 워싱턴에서 허리케인 샌디를 찾아낸 것과 같은 성격의 검증이다.

## 예측 밴드 시각화

최우수 모델(랜덤포레스트) 300개 트리의 예측 분포로 10~90 백분위(80%) 밴드를 만들어 2016년 실제값과 겹쳐 봤다(`outputs/model/prediction_band_2016.html`).

- 밴드 안에 들어간 날: **254일 / 365일 (69.6%)**
- 명목 80% 구간인데 실측 커버리지가 69.6%로 낮은 이유: 이 밴드는 트리 간 예측 분산(모델 불확실성)만 반영하고, 데이터 자체의 순수 잔차(residual noise)는 포함하지 않기 때문이다. 실무 적용 시 conformal prediction 등으로 보정이 필요하다는 한계를 리포트에 명시했다.

## 프로젝트 구조

```
02_london/
├── README.md                   이 파일
├── requirements.txt
├── data/
│   ├── preprocess/              원본(london_merged.csv) + 일별 변환본(day_london.csv)
│   └── processed/                연도별 분리본(day_london_2015.csv, day_london_2016.csv)
├── src/                         분석 파이프라인 (숫자 순서대로 실행)
│   ├── 01_hourly_to_daily.py     시간별 -> 일별 변환
│   ├── 02_split_by_year.py       연도별 분리
│   ├── 03_data_quality_check.py
│   ├── 04_eda_visualizations.py
│   ├── 05_linear_regression.py
│   ├── 06_random_forest.py
│   ├── 07_xgboost_baseline.py
│   ├── 08_ridge_lasso.py         Ridge/Lasso (정규화 선형회귀)
│   ├── 09_xgboost_tuning.py
│   ├── 10_model_comparison.py
│   ├── 11_feature_importance.py
│   ├── 12_shap_analysis.py
│   ├── 13_anomaly_detection.py
│   ├── 14_prediction_band.py     예측 밴드 vs 실제값 데이터 생성
│   └── common.py                 공통 설정 (입력 변수 목록, 최적 하이퍼파라미터)
├── outputs/
│   ├── quality/, eda/, model/, anomaly/
└── docs/
    ├── REPORT.md                  상세 분석 리포트 (최종 결론 포함)
    ├── CLAUDE_WORKFLOW.md         Claude Code 협업 과정 기록
    ├── ANALYSIS_PLAN.md           통합 분석 계획서 (실행 전 수립)
    ├── DAILY_CONVERSION_PLAN.md   시간별->일별 변환 계획·실행 결과
    └── PREPROCESSING_NOTES.md     컬럼별 집계 규칙 확정 근거
```

## 재현 방법

```bash
pip install -r requirements.txt

# data/preprocess/london_merged.csv(원본 시간별 데이터)가 있는 상태에서 순서대로 실행
python src/01_hourly_to_daily.py     # data/preprocess/day_london.csv 생성
python src/02_split_by_year.py       # data/processed/day_london_2015.csv, 2016.csv 생성
python src/03_data_quality_check.py
python src/04_eda_visualizations.py
python src/05_linear_regression.py
python src/06_random_forest.py
python src/07_xgboost_baseline.py
python src/08_ridge_lasso.py
python src/09_xgboost_tuning.py      # GridSearchCV, 수 분 소요
python src/10_model_comparison.py    # 08, 09 결과를 함께 파싱하므로 반드시 그 뒤에 실행
python src/11_feature_importance.py
python src/12_shap_analysis.py
python src/13_anomaly_detection.py
python src/14_prediction_band.py
```

## 데이터 출처

- [London bike sharing dataset (Kaggle, hmavrodiev)](https://www.kaggle.com/datasets/hmavrodiev/london-bike-sharing-dataset)
