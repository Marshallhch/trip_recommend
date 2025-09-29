import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from config import BUDGET_ORDER, DATA_DIR

def season_match(row_season, season: str) -> bool:
  """
  pois_korea의 season_suitability와 사용자의 season이 맞는지 여부
  """
  pass

def budget_ok(row_int: str, budget: str) -> bool:
  """예산 허용 여부"""
  pass

def score_poi(row, inferred_tags, user_interests, seaon, budget) -> float:
  """
    위치 점수 계산 - 사용자 관심, 추론태그, 계절, 예산 반영한 점수
  """
  pass

def rank_pois(pois_df: pd.DataFrame, inferred_tags, user_interests, season, budget):
  """
  위치에 점수를 부여하고 점수 순으로 정렬된 명소 데이터 반환
  """
  pass