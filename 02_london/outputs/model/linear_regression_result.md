# 선형회귀 분석 결과 (train=2015, test=2016)

- RMSE: 4359.46
- MAE: 3387.17
- R2: 0.7499

### 원 스케일 회귀계수
| 변수 | 계수 |
|---|---|
| workingday | 5390.92 |
| is_holiday | -1958.77 |
| t2 | 734.03 |
| weather_code | -431.81 |
| wind_speed | -352.76 |
| hum | -328.81 |
| weekday | 200.97 |
| mnth | -144.55 |
| season | -69.92 |

### 표준화 회귀계수 (단위를 맞춘 상대적 영향력)
| 변수 | 표준화 계수 |
|---|---|
| t2 | 4208.85 |
| hum | -3166.07 |
| workingday | 2479.43 |
| wind_speed | -2227.42 |
| weather_code | -1163.92 |
| mnth | -495.10 |
| weekday | 401.25 |
| is_holiday | -269.74 |
| season | -77.81 |
