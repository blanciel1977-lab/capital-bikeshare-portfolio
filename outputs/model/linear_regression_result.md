# 선형회귀 분석 결과
## (1) 데이터 누수 재현: casual/registered를 입력에 포함한 경우
- RMSE: 0.0000, R2: 1.0000
- cnt = casual + registered로 정의된 값이라, 두 컬럼을 입력에 넣으면 모델이 항등식을 그대로 학습해 R2가 사실상 1.0이 된다 (leakage).
- casual 계수: 1.0000, registered 계수: 1.0000

## (2) casual/registered 제외 후 재학습 (11개 변수)
- RMSE: 1166.02
- MAE: 863.86
- R2: 0.6131

### 원 스케일 회귀계수
| 변수 | 계수 |
|---|---|
| atemp | 6272.92 |
| windspeed | -2232.16 |
| yr | 2026.31 |
| hum | -1003.68 |
| temp | -614.89 |
| weathersit | -506.21 |
| holiday | -375.41 |
| season | 296.74 |
| workingday | 69.39 |
| weekday | 46.08 |
| mnth | 27.52 |

### 표준화 회귀계수 (단위를 맞춘 상대적 영향력)
| 변수 | 표준화 계수 |
|---|---|
| atemp | 1051.79 |
| yr | 980.98 |
| season | 307.96 |
| weathersit | -276.56 |
| windspeed | -169.91 |
| hum | -148.33 |
| temp | -115.68 |
| weekday | 92.36 |
| mnth | 88.82 |
| holiday | -61.28 |
| workingday | 32.24 |
