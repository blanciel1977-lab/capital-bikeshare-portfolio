# Capital Bikeshare 대여 수요 요인 분석

Capital Bikeshare(미국 워싱턴 D.C. 일대) 일별 자전거 대여 데이터를 사용해, **"어떤 요인이 자전거 대여 수요에 영향을 주는가"**를 데이터 확인부터 모델링, 모델 재사용성 검증까지 다룬 프로젝트다.

전체 분석 과정은 CLI 기반 AI 코딩 에이전트 **Claude Code**를 활용해 진행했다. 단순히 "AI가 분석해줬다"가 아니라, 데이터 확인 -> 정제 -> 탐색적 분석 -> 시각화 -> 모델링 -> 결과 검증 -> 보정 -> 문서화로 이어지는 실제 분석 워크플로우를 Claude Code CLI로 수행한 사례다. 이 과정 자체를 [docs/CLAUDE_WORKFLOW.md](docs/CLAUDE_WORKFLOW.md)에 별도로 정리했다.

## 핵심 결과 요약

- 데이터: 731일(2011-01-01~2012-12-31), 16개 컬럼, 결측치 0건
- 모델 비교 결과, **튜닝된 XGBoost가 RMSE 626.23 / R2 0.9022로 최우수**
- 초기 회귀분석에서 **데이터 누수(casual+registered=cnt)**를 발견해 제거
- 최우수 모델에서 `yr`(연도)의 중요도가 34%로 가장 높았으나, 이는 **미래 연도로 일반화 불가능한 정보**임을 확인 → 연도별 상대지수로 detrend해 **재사용 가능한 요인(계절·기온·날씨등급, 합산 82%)**을 별도로 도출
- 상세 내용은 [docs/REPORT.md](docs/REPORT.md) 참고

| 순위 | 모델 | RMSE | MAE | R2 |
|---|---|---|---|---|
| 1 | **XGBoost (튜닝 후)** | **626.23** | **428.57** | **0.9022** |
| 2 | XGBoost (기본값) | 655.49 | 446.11 | 0.8928 |
| 3 | 랜덤포레스트 | 669.10 | 429.28 | 0.8884 |
| 4 | 선형회귀 | 831.29 | 617.39 | 0.8277 |

(수치는 `pip install -r requirements.txt` 후 `src/` 스크립트를 순서대로 실행하면 그대로 재현된다. 라이브러리 버전에 따라 XGBoost 계열은 소수점 단위로 값이 달라질 수 있으나 모델 순위와 결론은 동일하다.)

![모델 비교](outputs/model/model_comparison.png)
![변수 중요도](outputs/model/xgboost_feature_importance.png)

## 프로젝트 구조

```
nyc-bikeshare-portfolio/
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
│   └── 10_detrend_analysis.py
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

# 2. 프로젝트 루트(nyc-bikeshare-portfolio/)에서 순서대로 실행
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
```

각 스크립트는 `outputs/` 아래 자신의 결과(md/png/csv)를 저장하며, 위 순서대로 실행하면 이 README와 `docs/REPORT.md`에 기록된 수치가 그대로 재현된다.

## 데이터 출처

- [Bike Sharing Dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) — Capital Bikeshare 일별/시간별 대여 데이터
