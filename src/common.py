"""
스크립트 공통 유틸 — 여러 단계가 공유하는 설정을 한 곳에서 관리한다.

1) 입력 변수 목록 (get_feature_cols)
2) 최적 하이퍼파라미터 (load_best_params)

각 스크립트가 값을 따로 하드코딩하면 조건을 바꿀 때 일부만 반영되어
결과가 서로 어긋나므로, 이 모듈을 단일 출처(single source of truth)로 삼는다.
"""
import ast
import re

TUNING_RESULT_PATH = 'outputs/model/xgboost_tuning_result.md'

# 목표변수 및 식별자
#   instant/dteday : 식별자·날짜 (예측에 쓰면 의미 없음)
#   cnt            : 목표변수
#   casual/registered : cnt = casual + registered 이므로 데이터 누수 (4단계 참고)
BASE_EXCLUDE = ['instant', 'dteday', 'cnt', 'casual', 'registered']

# 다중공선성으로 제외하는 변수
#   temp(기온)와 atemp(체감온도)의 상관계수는 0.9917로 사실상 같은 변수다.
#   둘을 모두 넣으면 트리가 매 분기마다 사실상 동전 던지기로 하나를 고르고,
#   gain 기반 중요도가 두 변수 사이에서 임의로 쪼개져 개별 순위를 신뢰할 수 없게 된다.
#   (실제로 메인 모델에서는 atemp > temp, detrend 모델에서는 temp > atemp로 순위가 뒤집혔다.)
#   자전거 이용 결정에 더 직접적인 체감온도(atemp)를 남기고 temp를 제외한다.
COLLINEAR_EXCLUDE = ['temp']


def get_feature_cols(df, extra_exclude=()):
    """입력 변수 컬럼 목록을 반환한다.

    extra_exclude: 특정 단계에서만 추가로 빼야 하는 컬럼
                   (예: 10단계 detrend는 yr, cnt_relative를 추가 제외)
    """
    exclude = set(BASE_EXCLUDE) | set(COLLINEAR_EXCLUDE) | set(extra_exclude)
    return [c for c in df.columns if c not in exclude]


def load_best_params():
    """튜닝 결과 md에서 최적 하이퍼파라미터 dict를 파싱해 반환한다."""
    try:
        text = open(TUNING_RESULT_PATH, encoding='utf-8').read()
    except FileNotFoundError:
        raise SystemExit(
            f'{TUNING_RESULT_PATH} 가 없습니다. src/07_xgboost_tuning.py 를 먼저 실행하세요.'
        )

    match = re.search(r'\{[^}]*\}', text)
    if not match:
        raise SystemExit(f'{TUNING_RESULT_PATH} 에서 하이퍼파라미터를 찾지 못했습니다.')

    params = ast.literal_eval(match.group(0))
    params.update(random_state=42, n_jobs=-1)
    return params
