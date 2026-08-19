"""
런던 프로젝트 공통 유틸 (01_washington/src/common.py와 같은 역할)
"""
import ast
import re

TUNING_RESULT_PATH = 'outputs/model/xgboost_tuning_result.md'

# ANALYSIS_PLAN.md 1절에서 확정한 최종 입력 변수
FEATURE_COLS = ['season', 'mnth', 'weekday', 'is_holiday', 'workingday',
                 'weather_code', 't2', 'hum', 'wind_speed']
TARGET_COL = 'cnt'


def get_feature_cols():
    return list(FEATURE_COLS)


def load_best_params():
    try:
        text = open(TUNING_RESULT_PATH, encoding='utf-8').read()
    except FileNotFoundError:
        raise SystemExit(
            f'{TUNING_RESULT_PATH} 가 없습니다. src/08_xgboost_tuning.py 를 먼저 실행하세요.'
        )
    match = re.search(r'\{[^}]*\}', text)
    if not match:
        raise SystemExit(f'{TUNING_RESULT_PATH} 에서 하이퍼파라미터를 찾지 못했습니다.')
    params = ast.literal_eval(match.group(0))
    params.update(random_state=42, n_jobs=-1)
    return params
