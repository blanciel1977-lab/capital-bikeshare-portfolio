# Capital Bikeshare 대여 수요 요인 분석

Capital Bikeshare(미국 워싱턴 D.C. 일대) 일별 자전거 대여 데이터를 사용해, **"어떤 요인이 자전거 대여 수요에 영향을 주는가"**를 데이터 확인부터 모델링, 모델 재사용성 검증까지 다룬 프로젝트다.

전체 분석 과정은 CLI 기반 AI 코딩 에이전트 **Claude Code**를 활용해 진행했다. 단순히 "AI가 분석해줬다"가 아니라, 데이터 확인 -> 정제 -> 탐색적 분석 -> 시각화 -> 모델링 -> 결과 검증 -> 보정 -> 문서화로 이어지는 실제 분석 워크플로우를 Claude Code CLI로 수행한 사례다. 이 과정 자체를 [docs/CLAUDE_WORKFLOW.md](docs/CLAUDE_WORKFLOW.md)에 별도로 정리했다.

## 핵심 결과 요약

- 데이터: 731일(2011-01-01~2012-12-31), 16개 컬럼, 결측치 0건
- **시간순 분할**로 평가 — 앞 80%(2011-01-01~2012-08-06) 학습, 뒤 20%(2012-08-07~2012-12-31) 예측
- 모델 비교 결과, **튜닝된 XGBoost가 RMSE 968.75 / R2 0.7329로 최우수** (같은 기간 평균 예측 baseline RMSE 2,561 대비 62% 개선)
- 초기 회귀분석에서 **데이터 누수(casual+registered=cnt)**를 발견해 제거
- 최우수 모델에서 `yr`(연도)의 중요도가 46%로 가장 높았으나, 이는 **미래 연도로 일반화 불가능한 정보**임을 확인 → 연도별 상대지수로 detrend해 **재사용 가능한 요인(체감온도·계절·날씨등급, 합산 78%)**을 별도로 도출
- `temp`~`atemp` 상관이 0.9917로 중복이라 중요도 순위가 불안정했던 문제를 확인하고 `temp`를 제외, **SHAP으로 영향의 방향과 비선형 형태까지** 분석
- 상세 내용은 [docs/REPORT.md](docs/REPORT.md) 참고

| 순위 | 모델 | RMSE | MAE | R2 |
|---|---|---|---|---|
| 1 | **XGBoost (튜닝 후)** | **968.75** | **770.37** | **0.7329** |
| 2 | XGBoost (기본값) | 1,040.61 | 850.31 | 0.6919 |
| 3 | 랜덤포레스트 | 1,124.18 | 903.22 | 0.6404 |
| 4 | 선형회귀 | 1,163.07 | 862.73 | 0.6151 |

## 이 프로젝트에서 가장 중요한 부분: 자기 검증과 수정

**초기 버전의 R2는 0.90이었다. 지금은 0.73이다. 점수가 내려간 것이 이 프로젝트의 성과다.**

모델을 한 번 돌려 좋은 숫자를 얻는 것보다, 그 숫자가 어떻게 만들어졌는지 되짚어 **믿을 수 있는 숫자로 바꾸는 과정**에 무게를 뒀다. 분석을 마친 뒤 스스로 코드 리뷰를 수행해 다음 문제들을 찾아내고 모두 수정했다.

| 발견한 문제 | 실제 영향 | 조치 |
|---|---|---|
| **데이터 누수** — `cnt = casual + registered` | R2가 정확히 1.0000 (예측이 아닌 항등식) | 두 변수를 입력에서 제외 |
| **무작위 분할** — 시계열인데 날짜를 섞음 | 오차를 **32% 낮게** 보고 (969건 → 662건) | 시간순 분할로 교체 |
| **`yr` 외삽 불가** — 2개 범주뿐인 변수 | 연도 간 예측 시 **R2 -0.53** (평균 예측보다 나쁨) | detrend 상대지수 모델로 분리 |
| **CV의 KFold 편향** — 튜닝 단계에 같은 결함 재발 | CV가 실제보다 **364건 후한** 점수 부여 | `TimeSeriesSplit`으로 교체 |
| **다중공선성** — `temp`~`atemp` 상관 0.9917 | 모델에 따라 중요도 **순위가 역전**됨 | `temp` 제외 + SHAP 도입 |
| **문서-코드 불일치** | 실제로 없는 실험이 문서에 서술됨 | 해당 서술 삭제 |

각 문제는 지적에 그치지 않고 **수치로 검증했다.** 예를 들어 분할 방식의 영향은 `src/11_split_comparison.py`로, CV 방식의 영향은 튜닝 결과에 두 방식을 나란히 기록해 확인할 수 있다. 남은 한계는 [REPORT 8절](docs/REPORT.md)에 숨기지 않고 명시했다.

### 평가 방식에 대하여

이 데이터는 일별 시계열이므로 **날짜를 무작위로 섞어 분할하지 않았다.** 무작위로 나누면 테스트 날짜의 바로 앞뒤 날이 학습셋에 포함되는데, 대여량과 날씨는 하루 사이에 거의 변하지 않으므로 모델이 "기온·계절로 수요를 추론"하는 대신 "양옆 값을 보고 가운데를 메우는" 쉬운 문제를 풀게 된다.

분할 방식만 바꿔 실제로 측정한 결과(`src/11_split_comparison.py`):

| 분할 방식 | RMSE | R2 |
|---|---|---|
| 무작위 분할 | 662.31 | 0.8906 |
| **시간순 분할 (채택)** | **968.75** | **0.7329** |
| 2011년 학습 → 2012년 예측 | 2,206.05 | -0.5253 |

무작위 분할은 실제 오차(969건)를 662건으로 **32% 낮게** 보고한다. 세 번째 행의 음수 R2는 `yr` 변수가 미래 연도로 외삽되지 않는다는 것을 보여주며, 이것이 아래 detrend 분석의 출발점이다.

같은 이유로 **하이퍼파라미터 튜닝의 교차검증도 `TimeSeriesSplit`을 사용했다.** 기본 `KFold`로 채점하면 동일 설정에 대해 RMSE 605가 나오는데, 실제 test는 969다. 이 후한 점수를 기준으로 조합을 고르면 선택 자체가 달라진다(실제로 `max_depth` 4→3, `learning_rate` 0.05→0.1로 바뀜).

(수치는 `pip install -r requirements.txt` 후 `src/` 스크립트를 순서대로 실행하면 그대로 재현된다. 라이브러리 버전에 따라 XGBoost 계열은 소수점 단위로 값이 달라질 수 있으나 모델 순위와 결론은 동일하다.)

![모델 비교](outputs/model/model_comparison.png)
![변수 중요도](outputs/model/xgboost_feature_importance.png)

### 요인의 크기뿐 아니라 방향과 형태까지

gain 중요도는 "얼마나 중요한가"만 알려주므로, SHAP으로 "값이 커지면 수요가 늘어나는가, 어느 구간에서 꺾이는가"까지 분석했다(`src/12_shap_analysis.py`).

![SHAP 요약](outputs/model/shap_beeswarm.png)

체감온도·계절은 양(+), 습도·풍속·악천후는 음(-) 방향으로 작용한다. 또한 두 지표의 순위가 최대 3계단까지 달랐는데, 특히 `weathersit`은 gain에서 3위지만 SHAP 기준으로는 6위다.

![체감온도 의존성](outputs/model/shap_dependence_atemp.png)

기온 효과는 선형이 아니다. 체감온도가 오를수록 수요가 급격히 늘지만 **0.5~0.6 구간에서 정점을 찍고 그 위로는 더 늘지 않는다.** 선형회귀가 이 데이터에서 성능이 낮았던 이유가 이 곡선으로 설명된다.

## 프로젝트 구조

```
capital-bikeshare-portfolio/
├── README.md                  이 파일
├── requirements.txt            의존성 목록
├── data/
│   └── day.csv                 원본 데이터 (train/test는 03번 스크립트 실행 시 생성됨)
├── src/                        분석 파이프라인 (숫자 순서대로 실행)
│   ├── 01_data_quality_check.py
│   ├── 02_eda_visualizations.py
│   ├── 03_train_test_split.py
│   ├── 04_linear_regression.py
│   ├── 05_random_forest.py
│   ├── 06_xgboost_baseline.py
│   ├── 07_xgboost_tuning.py
│   ├── 08_model_comparison.py
│   ├── 09_feature_importance.py
│   ├── 10_detrend_analysis.py
│   ├── 11_split_comparison.py   분할 방식이 평가에 미치는 영향 비교
│   ├── 12_shap_analysis.py      SHAP: 영향의 방향·비선형 형태 분석
│   └── common.py                공통 설정 (입력 변수 목록, 튜닝 파라미터)
├── outputs/
│   ├── quality/                 데이터 품질 점검 결과
│   ├── eda/                     탐색적 분석 시각화
│   └── model/                   모델 성능·중요도 결과
└── docs/
    ├── REPORT.md                상세 분석 리포트 (최종 결론 포함)
    └── CLAUDE_WORKFLOW.md       Claude Code를 활용한 분석 과정 정리
```

## 재현 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 프로젝트 루트(capital-bikeshare-portfolio/)에서 순서대로 실행
python src/01_data_quality_check.py
python src/02_eda_visualizations.py
python src/03_train_test_split.py     # data/train.csv, data/test.csv 생성
python src/04_linear_regression.py
python src/05_random_forest.py
python src/06_xgboost_baseline.py
python src/07_xgboost_tuning.py       # GridSearchCV, 수 분 소요
python src/08_model_comparison.py     # 04~07 결과 md를 파싱해 비교 그래프 생성
python src/09_feature_importance.py
python src/10_detrend_analysis.py
python src/11_split_comparison.py
python src/12_shap_analysis.py
```

각 스크립트는 `outputs/` 아래 자신의 결과(md/png/csv)를 저장하며, 위 순서대로 실행하면 이 README와 `docs/REPORT.md`에 기록된 수치가 그대로 재현된다.

## 데이터 출처

- [Bike Sharing Dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) — Capital Bikeshare 일별/시간별 대여 데이터
