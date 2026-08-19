# 선형회귀 분석 결과
## (1) 데이터 누수 재현: casual/registered를 입력에 포함한 경우
- RMSE: 0.0000, R2: 1.0000
- cnt = casual + registered로 정의된 값이라, 두 컬럼을 입력에 넣으면 모델이 항등식을 그대로 학습해 R2가 사실상 1.0이 된다 (leakage).
- casual 계수: 1.0000, registered 계수: 1.0000

## (2) casual/registered 제외 후 재학습 (11개 변수)
- RMSE: 1163.07
- MAE: 862.73
- R2: 0.6151

### 원 스케일 회귀계수
| 변수 | 계수 |
|---|---|
| atemp | 5581.10 |
| windspeed | -2258.81 |
| yr | 2027.18 |
| hum | -994.02 |
| weathersit | -508.04 |
| holiday | -380.16 |
| season | 295.46 |
| workingday | 69.93 |
| weekday | 46.02 |
| mnth | 27.97 |

### 표준화 회귀계수 (단위를 맞춘 상대적 영향력)
| 변수 | 표준화 계수 |
|---|---|
| yr | 981.41 |
| atemp | 935.79 |
| season | 306.63 |
| weathersit | -277.56 |
| windspeed | -171.93 |
| hum | -146.91 |
| weekday | 92.24 |
| mnth | 90.26 |
| holiday | -62.06 |
| workingday | 32.48 |
