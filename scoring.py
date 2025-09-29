import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from config import BUDGET_ORDER, DATA_DIR

# 명소 데이터 로드
poi_korea = pd.read_csv(DATA_DIR / 'pois_korea.csv')

def season_match(row_season, season: str) -> bool:
  """
  pois_korea의 season_suitability와 사용자의 season이 맞는지 여부
  Args:
    row_seasons: 계절 리스
    season: 사용자 계절
  Returns:
    bool: 사용자 계정과 데이터 계절 일치 여부
  """
  if pd.isna(row_season): # 명소 계절이 없으면 무조건 True
    return True
  sset = set(str(row_season).split("|"))
  return (season in sset) or ('사계절' in sset)

# print(season_match(poi_korea['season_suitability'][0], '겨울'))

def budget_ok(row_hint: str, budget: str) -> bool:
  """
  예산 허용 여부
  Args: 
    row_hint: 명소 예산 힌트
    budget: 사용자 예산
  Returns:
    bool: 허용 여부
  """
  # 명소 예산 힌트가 사용자 예산보다 낮거나 같으면 True
  return BUDGET_ORDER.get(row_hint, 1) <= BUDGET_ORDER.get(budget, 1)

# print(budget_ok('중', "상"))

def score_poi(row, inferred_tags, user_interests, season, budget) -> float:
  """
    위치 점수 계산 - 사용자 관심, 추론태그, 계절, 예산 반영한 점수
    Args:
      row: 명소 데이터
      inferred_tags: 추론 테그(관심사)
      user_interests: 사용자 관심사
      season: 사용자 계절
      budget: 사용자 예산
    Returns:
      float: 위치 점수
  """
  
  tags = set(str(row["tags"]).split("|"))
  overlap_inferred = len(tags & set(inferred_tags)) # 추론 태그하고 명소 태그하고 중복되는 개수
  overlap_user = len(tags & set(user_interests)) # 사용자 관심 태그와 명소 태그 중복 개수
  s_bonus = 1 if season_match(row['season_suitability'], season) else 0
  b_bonus = 1 if budget_ok(row["budget_hint"], budget) else 0

  # print(overlap_user, overlap_inferred, s_bonus, b_bonus)
  # 사용자가 명시한 관심사는 가장 강한 신호로 판단하여 가중치를 높게 둠
  # 추론 태그는 사용자 관심사에 비해 낮은 가중치를 둠
  # 결과에 계절 적합성이 있으면 1점 부여
  # 결과에 예산 적합성이 있으면 1점 부여
  return (2 * overlap_user) + (1.5 * overlap_inferred) + s_bonus + b_bonus

# print(score_poi(poi_korea.iloc[0], ["culture", "history"], ["technology", "photography"], "봄", "중"))
# (2*2) + (1.5*1) + 1 + 1 = 9.0

# 파이썬 은행가 반올림 법칙: https://www.quora.com/Why-do-programming-languages-round-1-5-and-2-5-both-to-2
# 판다스 데이터 프레임을 읽어서 적용하면 출력 포멧팅에서 은행가 반올림으로 처리됨. 수치의 높낮이를 평가하기 때문에 여기서는 그냥 진행
# 은행가 반올림은 짝수에 수렴: 7.5 -> 7, 8.5 -> 8
# print(score_poi(poi_korea.iloc[0], ['walking', 'shopping'], ['architecture', 'culture'], '여름', '하')) # (2 * 0) + (1.5 * 1) + 0 + 0 = 2
# print((2 * 0) + (1.5 * 1) + 0 + 0) # 1.5

def rank_pois(pois_df: pd.DataFrame, inferred_tags, user_interests, season, budget):
  """
  위치에 점수를 부여하고 점수 순으로 정렬된 명소 데이터 반환
  Args:
    pois_df: 명소 데이터
    inferred_tags: 추론 태그
    user_interests: 사용자 관심
    season: 사용자 계절
    budget: 사용자 예산
  Returns:
    ranked: 점수 순으로 정렬된 명소 데이터
  """
  pois = pois_df.copy() # 기존 데이터를 가공해야 하기 때문에 복사해서 사용
  pois["score"] = pois.apply(lambda r: score_poi(r, inferred_tags, user_interests, season, budget), axis=1)
  ranked = pois.sort_values(["score", "city"], ascending=False).reset_index(drop=True) # 인덱스 초기화
  return ranked

# print(rank_pois(poi_korea, ["warking", "shoping"], ["architecture", 'culture'], '여름', '중'))