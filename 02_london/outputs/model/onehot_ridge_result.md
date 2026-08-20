# 원-핫 인코딩 + Ridge 재실험 (train=2015, test=2016)

`season`, `weather_code`를 원-핫 인코딩(10개 더미: season_0.0, season_1.0, season_2.0, season_3.0, weather_code_1.0, weather_code_2.0, weather_code_3.0, weather_code_4.0, weather_code_7.0, weather_code_26.0), `t2/hum/wind_speed`는 표준화, 나머지는 그대로 사용.

weather_code=26.0 표본 수: train 1일, test 2일 (극희귀 범주)

## (1) 무규제 선형회귀 (원-핫 인코딩)
- RMSE: 4247.43
- MAE: 3178.67
- R2: 0.7626
- `weather_code_26.0` 계수: **+2453.1** (표본 1일뿐인 희귀 범주 — 불안정 여부 확인용)

## (2) Ridge (원-핫 인코딩, TimeSeriesSplit으로 alpha 탐색)
- 선택된 alpha: 5.1507
- RMSE: 4225.45
- MAE: 3181.63
- R2: 0.7650
- `weather_code_26.0` 계수: **+460.1** (무규제 대비 안정화 정도 확인)

### 표준화 회귀계수 (Ridge)
| 변수 | 계수 |
|---|---|
| workingday | 5090.03 |
| t2 | 3528.37 |
| hum | -3004.24 |
| weather_code_7.0 | -2432.24 |
| season_3.0 | -1841.62 |
| wind_speed | -1805.19 |
| season_1.0 | 1433.57 |
| season_2.0 | 1423.42 |
| season_0.0 | -1015.38 |
| weather_code_1.0 | 829.70 |
| weather_code_2.0 | 752.11 |
| weather_code_3.0 | 507.12 |
| weather_code_26.0 | 460.10 |
| is_holiday | -428.88 |
| mnth | -263.04 |
| weekday | 212.46 |
| weather_code_4.0 | -116.80 |

## 기존 결과와 비교

| 모델 | 인코딩 | RMSE | R2 |
|---|---|---|---|
| 랜덤포레스트 (기존 최우수) | 서수 | 4,308.82 | 0.7556 |
| Ridge (기존) | 서수 | 4,357.40 | 0.7501 |
| 선형회귀 (원-핫, 무규제) | 원-핫 | 4,247.43 | 0.7626 |
| **Ridge (원-핫)** | 원-핫 | **4,225.45** | **0.7650** |

**원-핫 Ridge가 랜덤포레스트를 이겼다** (RMSE 4,225.45 vs 4,308.82).
