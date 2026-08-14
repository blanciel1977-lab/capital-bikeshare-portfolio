# 선형회귀 분석 결과
## (1) 데이터 누수 재현: casual/registered를 입력에 포함한 경우
- RMSE: 0.0000, R2: 1.0000
- cnt = casual + registered로 정의된 값이라, 두 컬럼을 입력에 넣으면 모델이 항등식을 그대로 학습해 R2가 사실상 1.0이 된다 (leakage).
- casual 계수: 1.0000, registered 계수: 1.0000

## (2) casual/registered 제외 후 재학습 (11개 변수)
- RMSE: 831.29
- MAE: 617.39
- R2: 0.8277

### 원 스케일 회귀계수
| 변수 | 계수 |
|---|---|
| atemp | 3488.04 |
| temp | 2097.25 |
| windspeed | -2080.54 |
| yr | 2024.00 |
| hum | -865.44 |
| weathersit | -632.86 |
| season | 524.72 |
| holiday | -391.55 |
| workingday | 160.80 |
| weekday | 72.94 |
| mnth | -38.44 |

### 표준화 회귀계수 (단위를 맞춘 상대적 영향력)
| 변수 | 표준화 계수 |
|---|---|
| yr | 1011.41 |
| season | 578.57 |
| atemp | 563.42 |
| temp | 380.43 |
| weathersit | -347.85 |
| windspeed | -160.92 |
| weekday | 147.67 |
| mnth | -131.72 |
| hum | -122.93 |
| workingday | 75.64 |
| holiday | -61.94 |
