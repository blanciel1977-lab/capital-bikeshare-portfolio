# Claude Code 데이터 분석 포트폴리오

**터미널 기반 AI 코딩 에이전트 Claude Code**를 데이터 분석 파트너로 사용해, 서로 다른 두 도시의 자전거 공유 데이터를 각각 처음부터 끝까지 분석한 프로젝트 모음이다. 단순히 "AI가 분석해줬다"가 아니라, 데이터 확인 → 전처리 → EDA → 모델링 → 검증 → 보정 → 문서화로 이어지는 실제 분석 워크플로우를 Claude Code CLI로 수행한 기록이다.

두 프로젝트는 독립적으로 진행됐지만, **워싱턴에서 배운 방법론적 교훈을 런던 분석 설계 단계에서부터 그대로 적용**했다는 점에서 이어져 있다. 이 "교훈의 전이" 과정 자체가 이 포트폴리오에서 가장 보여주고 싶은 부분이다.

## 프로젝트 목록

### [01_washington](01_washington/README.md) — Capital Bikeshare (워싱턴 D.C.)

일별 대여 데이터로 수요 요인을 분석. **가장 중요한 결과는 성능 수치 자체가 아니라, 자기 검증을 통해 초기 R²를 0.90에서 0.73으로 스스로 낮춘 과정**이다.

- 데이터 누수(`cnt = casual + registered`), 무작위 분할로 인한 성능 과대평가(32%), 튜닝 단계 CV의 동일 결함 재발, `temp`~`atemp` 다중공선성을 순서대로 발견·수정
- 최종 모델: 튜닝된 XGBoost (RMSE 968.75, R² 0.7329)
- `yr`(연도) 변수가 미래로 외삽 불가능함을 실측(R² -0.53)으로 확인 → detrend 재구성으로 재사용 가능한 요인만 별도 도출
- 사후 분석: Isolation Forest로 이상치 탐지 → 상위권 대부분이 실제 허리케인 샌디·폭설 등 기록된 기상 사건과 정확히 일치함을 외부 검색으로 검증

### [02_london](02_london/README.md) — London Bike Sharing (2015 학습 → 2016 예측)

시간별 원본을 일별로 직접 변환하는 단계부터 시작해, 2015년으로 학습한 모델로 2016년 대여량을 예측.

- 워싱턴에서 배운 교훈을 계획 단계부터 선반영: `yr`은 애초에 입력에서 제외(사후 보정 불필요), `t1`~`t2` 다중공선성 사전 확인 후 제외, 파생 규칙을 문서화해 다른 기간 데이터에도 재사용 가능하게 설계
- 최종 모델: 랜덤포레스트 (RMSE 4,308.82, R² 0.7556) — 워싱턴은 항상 XGBoost가 1위였지만, 이번엔 가정 없이 실측으로 다른 모델을 채택
- 이상치 탐지에서 발견한 2015-07-09(대여량 최고치)가 **런던 지하철 전면파업일**과 정확히 일치함을 외부 뉴스로 검증
- 예측 밴드(랜덤포레스트 300개 트리의 10~90 백분위) vs 2016년 실제값 시각화로 모델의 불확실성을 정직하게 노출(커버리지 69.6%, 한계로 명시)

## 이 포트폴리오에서 확인할 수 있는 것

| 역량 | 확인 위치 |
|---|---|
| 데이터 품질 점검·이상치 탐지 | 각 프로젝트 `outputs/quality/`, `outputs/anomaly/` |
| 평가 설계의 함정(무작위 분할, KFold vs TimeSeriesSplit)을 실측으로 잡아내는 능력 | `01_washington/docs/REPORT.md` 4.2~4.4절 |
| 한 프로젝트의 교훈을 다음 프로젝트 설계에 선반영하는 능력 | `02_london/docs/ANALYSIS_PLAN.md` 1절 |
| 모델 결과를 외부 사실(실제 기상·사회 이벤트)과 대조해 검증하는 습관 | 두 프로젝트 REPORT.md의 이상치 탐지 절 |
| Claude Code와의 실제 협업 과정(질문-검증-수정 사이클) | 각 프로젝트 `docs/CLAUDE_WORKFLOW.md` |
| 세션 간 연속성을 위한 작업 기록 습관 | [`worklog/`](worklog/) — 날짜별 작업 로그 |

## 폴더 구조

```
.
├── README.md                  이 파일 (포트폴리오 개요)
├── 01_washington/              Capital Bikeshare 분석 (완료)
│   ├── README.md               프로젝트 요약
│   ├── docs/
│   │   ├── REPORT.md            상세 분석 리포트
│   │   ├── CLAUDE_WORKFLOW.md   Claude Code 협업 과정 기록
│   │   └── HOURLY_TO_DAILY_RULES.md   시간별→일별 변환 규칙서 (런던에 재사용됨)
│   ├── src/                    01~12단계 분석 파이프라인
│   ├── data/, outputs/
├── 02_london/                  London Bike Sharing 분석 (완료)
│   ├── README.md               프로젝트 요약
│   ├── docs/
│   │   ├── REPORT.md            상세 분석 리포트
│   │   ├── CLAUDE_WORKFLOW.md   Claude Code 협업 과정 기록
│   │   ├── ANALYSIS_PLAN.md     통합 분석 계획서 (실행 전 수립)
│   │   ├── DAILY_CONVERSION_PLAN.md, PREPROCESSING_NOTES.md
│   ├── src/                    01~13단계 분석 파이프라인
│   ├── data/, outputs/
└── worklog/                    날짜별 작업 로그 (YYYY-MM-DD.md)
```

## 재현 방법

각 프로젝트 폴더의 README.md를 참고. 두 프로젝트 모두 `pip install -r requirements.txt` 후 `src/` 스크립트를 번호 순서대로 실행하면 문서에 기록된 수치가 그대로 재현된다.
