# 런던 데이터셋 작업 워크플로우

`01_washington/docs/CLAUDE_WORKFLOW.md`와 같은 성격의 문서로, 런던(Capital Bikeshare가 아닌 London bike sharing) 데이터셋에 대해 진행한 과정을 단계별로 기록한다. 원칙은 워싱턴 프로젝트와 동일: **원본 데이터는 임의로 수정하지 않고, 애매한 지점은 진행 전 사용자에게 먼저 확인한다.**

## 1. 원본 데이터 확인 (2026-08-19)

- `02_london/data/raw/london_merged.csv` 로드 후 행/열 수, 컬럼·자료형, 결측치, 기간을 조회
- 결과: 17,414행 × 10컬럼, 결측치 0건, 기간 2015-01-04~2017-01-03(731일, 시간별 데이터)
- 컬럼: `timestamp, cnt, t1, t2, hum, wind_speed, weather_code, is_holiday, is_weekend, season` — 워싱턴과 컬럼명·인코딩 체계가 다름을 확인(예: weather_code 7종 범주 vs 워싱턴 weathersit 1~4)

## 2. 워싱턴 변환 규칙서와의 대조

- `HOURLY_TO_DAILY_RULES.md`를 런던에 적용하기 전, 컬럼을 3가지로 분류해 표로 정리:
  1. 규칙 그대로 사용 가능: `cnt`(합계), `t1`/`t2`(평균), `season`/`is_holiday`/`is_weekend`(고정값) — 730일 전부 하루 내 고정값임을 실측 확인
  2. 규칙 그대로 쓰면 안 됨: `weather_code`(서수 가정 미검증), `hum`/`wind_speed`(스케일이 워싱턴과 달라 반올림 규칙 재조정 필요), 결측 처리 규칙(서머타임으로 인한 34일의 23/25시간을 진짜 결측과 구분해야 함)
  3. 아예 없어서 새로 만들어야 함: `dteday`/`yr`/`mnth`/`weekday`(timestamp에서 파생 가능), `workingday`(정의 재확인 후 파생), `casual`/`registered`(**파생 불가** — 런던 원본에 사용자 유형 구분 자체가 없음)

## 3. weather_code / season 정의 조사 및 결정

- 사용자 요청으로 `weather_code`(1,2,3,4,7,10,26)와 `season`(0~3)의 실제 의미를 웹 검색으로 확인 (Kaggle 공식 데이터셋 hmavrodiev/london-bike-sharing-dataset 문서 기준)
  - season: 0=spring, 1=summer, 2=fall, 3=winter
  - weather_code: 1=Clear, 2=구름조금, 3=구름많음, 4=흐림, 7=비, 10=뇌우, 26=눈 — 이 순서가 실제 악화 순서와 일치함을 확인
- 사용자가 두 가지를 확정: **(1) weather_code는 원본 코드값을 그대로 쓰고 워싱턴 체계로 리매핑하지 않는다** (다른 기간 데이터가 추가로 들어와도 전처리를 최소화하기 위함), **(2) 집계 규칙(연산 방식)은 워싱턴과 동일하게 유지한다** — 즉 "최악등급 4시간 이상→그 값, 아니면 최빈값" 알고리즘은 그대로 쓰되, 리매핑 없이 원본 코드 위에서 바로 적용
- 결정 사항을 `02_london/docs/PREPROCESSING_NOTES.md`에 원칙 문서로 정리

## 4. 일별 변환 실행

- 실행 전 `02_london/docs/DAILY_CONVERSION_PLAN.md`로 계획서를 먼저 작성: 컬럼별 규칙, `n_hours_observed`(관측 시간 수) 컬럼 신설, `timestamp`에서 `yr`/`mnth`/`weekday` 파생, `workingday`(주말도 휴일도 아니면 1) 정의를 명시
  - `weekday` 인코딩은 워싱턴 day.csv 731일과 대조해 0=일요일~6=토요일임을 먼저 확인한 뒤 동일하게 맞춤
  - `workingday = (is_weekend==0) & (is_holiday==0)` 정의도 워싱턴 731일 전체와 실측 대조해 정확히 일치함을 사전 확인
  - `hum`/`wind_speed` 등 평균형 컬럼의 반올림은 소수 2자리로 확정(원본 정밀도 대비 과·과소 정밀 아님)
- `02_london/src/01_hourly_to_daily.py` 작성·실행 → `02_london/data/processed/day_london.csv` 생성
- 결과: 원본 17,414행(시간별) → **730행**(일별), **24시간 미만인 날 34일**
- 실행 중 예상 못 한 문제 발견: **`2016-09-02`가 원본에 아예 1행도 없이 통째로 빠져 있어** 일별 결과에도 행 자체가 없음(731일 기대 대비 730행인 이유). 계획서 "실행 결과" 절에 별도로 기록

## 5. 연도별 파일 분리

- `day_london.csv`(730행)를 `yr` 기준으로 2015년(362행, 01-04~12-31)/2016년(365행, 01-01~12-31)으로 나눠 각각 `day_london_2015.csv`, `day_london_2016.csv`로 저장. 2017년(3행, 01-01~01-03)은 요청대로 제외
- 두 파일 다 신규 생성이며 `day_london.csv` 원본은 그대로 둠
- 이후 `02_london/src/02_split_by_year.py`로 스크립트화해 저장(처음엔 1회성 코드로 실행했던 것을 재현 가능하게 정리)

## 6. 폴더 재배치에 따른 경로 정정

- 사용자가 `02_london/data/raw/` 폴더를 `data/preprocess/`로 수동 재배치(그 안에 `london_merged.csv`와 `day_london.csv`가 함께 위치), `data/processed/`에는 연도별 분리본만 남김
- 이로 인해 어긋난 경로를 일괄 정정: `01_hourly_to_daily.py`의 `RAW_PATH`/`OUT_PATH`를 `data/preprocess/`로 수정, `DAILY_CONVERSION_PLAN.md`·`PREPROCESSING_NOTES.md`의 입출력 경로 표기도 갱신
- **현재(최신) 경로**: 원본·일별 변환본 = `data/preprocess/`, 연도별 분리본 = `data/processed/`. 위 2~5절의 `data/raw/`, `data/processed/day_london.csv` 표기는 **그 시점 기준 과거 기록**이므로 지금 경로와는 다름에 유의

## 7. 통합 분석 계획 수립

- 사용자가 `01_washington/`의 전체 파이프라인(전처리→EDA→회귀분석→이상치탐지→리포트→워크플로우)을 참고해, 런던 데이터로 "2015년 학습 → 2016년 예측" 회귀분석을 위한 통합 계획을 요청
- `01_washington/docs/REPORT.md` 전체를 다시 읽어 방법론(시간순 분할 이유, 데이터 누수 검증, TimeSeriesSplit 튜닝, 다중공선성 처리, SHAP, detrend)을 재확인
- train(2015)/test(2016) 실측 통계를 미리 계산해 워싱턴에서 배운 교훈을 계획 단계부터 반영: `yr`은 train에서 상수·test에서 미지값이라 처음부터 입력 제외, `t1`~`t2` 상관 0.992로 `t1` 제외, `is_weekend`는 `weekday`에서 완전히 파생되는 중복 컬럼이라 제외 확인
- `02_london/docs/ANALYSIS_PLAN.md`로 저장: 최종 입력변수 9개, 스크립트 12단계 구성안, EDA/모델링/이상치탐지/리포트 절 구성을 워싱턴과 대응시켜 정리. 아직 실행 코드는 작성하지 않음(계획 단계)

## 8. 통합 계획 실행 — 전체 파이프라인 완주

`ANALYSIS_PLAN.md`에 따라 실행 전 3가지 미결 사항(season~mnth 다중공선성, 튜닝 그리드, contamination 비율)을 먼저 질문해 확인(모두 "워싱턴과 동일하게" 채택)한 뒤, `03`~`12` 스크립트를 순서대로 작성·실행했다.

- **03 품질점검**: train/test 결측 0건, 24시간 미만 관측일 train 18/test 16일 확인
- **04 EDA**: `t2`(체감기온)~`cnt` 상관 0.626으로 워싱턴과 유사하나, **`hum`(습도)~`cnt` 상관이 -0.587로 워싱턴(-0.101)보다 훨씬 강함**을 발견 — 리포트의 핵심 차별점으로 기록
- **05~08 모델 비교**: 선형회귀/랜덤포레스트/XGBoost(기본)/XGBoost(튜닝) 4개 비교 결과 **랜덤포레스트가 최우수**(RMSE 4,308.82, R2 0.7556) — 워싱턴에서는 항상 XGBoost(튜닝)가 1위였는데 이번엔 아니었음. 결과를 미리 가정하지 않고 실측대로 채택
  - matplotlib 한글 폰트 미설정으로 첫 EDA 실행 시 글자 깨짐 경고 발생 → `Malgun Gothic` 폰트 지정으로 해결
  - `weather_code` boxplot에서 float형과 order 리스트 길이 불일치 오류 발생 → `int` 캐스팅 및 실제 존재하는 값만 필터링해 해결
- **09 모델비교 시각화**: baseline(train 평균 예측) RMSE 8,757.83 대비 최우수 모델 50.8% 개선 확인
- **10~11 변수중요도·SHAP**: 워싱턴은 XGBoost 기준이었으나 이번엔 **최우수 모델(랜덤포레스트) 기준**으로 계산(스크립트를 하드코딩된 XGBoost가 아니라 실제 승자 모델을 쓰도록 설계). `t2`+`hum` 합산 65.4%로 지배적. SHAP과 importance 순위 차이는 평균 0.4계단(워싱턴 1.8계단보다 훨씬 작음) — `workingday`/`hum` 2·3위만 순서가 바뀜
- **12 이상치탐지**: 2015+2016 전체(727일)에서 37건(5.09%) 탐지. 상위권 대부분 눈/강풍/고습도가 겹친 겨울날 — 워싱턴 패턴과 일치. **6위(2015-07-09, 대여량 72,504건)를 외부 뉴스 검색으로 대조한 결과 런던 지하철 전면파업일과 정확히 일치**(당시 서비스 개시 이래 최다 이용일 73,094건 기록, Cycling Weekly 보도) — 워싱턴의 허리케인 샌디 사례와 같은 성격의 외부 검증 성공 사례

## 9. 최종 리포트 작성

`02_london/docs/REPORT.md`를 워싱턴 REPORT.md와 동일한 8절 구조(데이터개요/품질점검/EDA/모델링/이상치탐지/결론/검증이력/한계)로 작성. 워싱턴과 다르게 나온 부분(최우수 모델이 XGBoost가 아닌 점, 습도 영향이 더 큰 점)을 숨기지 않고 "차이점"으로 명시했다.

## 10. Ridge / Lasso 추가 및 스크립트 재정렬

- 사용자가 "선형회귀 종류를 몇 가지 했나"라고 질문 → 단순 OLS 1종(스케일만 원본/표준화 2가지)이었음을 확인·답변한 뒤, "Ridge/Lasso도 추가로 돌려볼 수 있나" 질문에 실제로 실행
- `RidgeCV`/`LassoCV` + TimeSeriesSplit(train 2015 내부에서만, XGBoost 튜닝과 동일 원칙)로 alpha 탐색. 결과: Ridge RMSE 4,357.40/R² 0.7501, Lasso RMSE 4,351.50/R² 0.7508 — 둘 다 일반 선형회귀보다 근소 개선되었으나 랜덤포레스트에는 못 미침. Lasso가 0으로 수렴시킨 변수는 없어 정규화 효과가 크지 않음(원래 과적합이 아니었다는 뜻)을 확인
- 사용자가 "반영"을 요청 → `10_model_comparison.py`가 Ridge/Lasso 결과까지 파싱하도록 수정, `model_comparison.png`/`model_comparison_summary.md` 재생성(6개 모델 비교로 확장)
- **스크립트 실행 순서 정합성을 위해 번호를 재정렬**: 신규 스크립트는 `10_model_comparison.py`(모델 비교, Ridge/Lasso 포함해 파싱)보다 먼저 실행돼야 하므로, `14_ridge_lasso.py`로 만들었던 파일을 `08_ridge_lasso.py`로 옮기고 이후 08~13번을 전부 한 칸씩 밀었다(09_xgboost_tuning ~ 14_prediction_band). 각 파일 docstring의 "N단계" 표기도 함께 수정하고, 재배치 후 08~11번을 재실행해 결과가 그대로 재현됨을 확인
- `README.md`, `REPORT.md`의 모델 비교 표와 스크립트 목록·재현 방법을 6개 모델/새 번호 기준으로 갱신

## 다음 단계 (미착수)

- weather_code 최빈값 동률 처리("더 나쁜 코드 우선")가 실제로 몇 건 발생했는지 아직 집계 안 함(변환 단계 이월 항목)
- 이상치 vs 예측오차 Top10 교집합 비교(워싱턴에서 수행했던 것)는 이번엔 진행 안 함 — REPORT 8절 한계로 명시
- 이번 재정렬·Ridge/Lasso 추가분 git commit/push 아직 안 함 (기존 `london_bike` 저장소도 재동기화 필요)
