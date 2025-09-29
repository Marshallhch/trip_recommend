from pathlib import Path

# 루트 디렉토리
ROOT = Path(__file__).resolve().parent

# 데이터 경로
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

# 하이퍼 파라미터
DEFAULT_PER_DAY_HOURS = 6 # 하루 일정 기본 값(6시간)
INTEREST_THRESHOLD = 0.35 # 다중 레이블 확률 임계값 - 0.35 이상일 경우 관심 있음
BUDGET_ORDER = {"하": 0, "중": 1, "상": 2} # 예산 범위
SEASONS_ALL = ["봄", "여름", "가을", "겨울"]