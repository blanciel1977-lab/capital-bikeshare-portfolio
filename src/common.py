"""
스크립트 공통 유틸.

07단계 튜닝 결과(outputs/model/xgboost_tuning_result.md)에 기록된 최적
하이퍼파라미터를 읽어온다. 09·10단계가 값을 각자 하드코딩하면 재튜닝 시
서로 어긋나므로, 튜닝 결과 파일을 단일 출처(single source of truth)로 삼는다.
"""
import ast
import re

TUNING_RESULT_PATH = 'outputs/model/xgboost_tuning_result.md'


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
