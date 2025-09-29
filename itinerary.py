from datetime import datetime, timedelta # timedelta: 특정 시점부터의 시간 간격
import pandas as pd
from scoring import season_match, budget_ok

def generate_itinerary(ranked_pois: pd.DataFrame, acts_df: pd.DataFrame, days: int, season: str, budget: str, city_hint: str=None, per_day_hours: int=6):
  """
  N일 코스 생성
  Args:
    ranked_pois: 평가된 poi 데이터 프레임
    acts_df: 활동 데이터 프레임
    days: 여행 일수
    season: 계절
    budget: 예산
    city_hint: 도시 힌트
    per_day_hours: 하루 여행 시간
  Returns:
    plan: 코스 데이터 프레임
  """

  # 변수 초기화
  plan = []
  start_date = datetime.now().date() # 현재시간
  used_pois = set() # 이미 사용한(방문한) 위치 목록

  # acts_df = [
  #   {"poi_id": 1, "activity": "전통시장 투어", "when": "오전", "est_cost_krw": 10000},
  #   {"poi_id": 1, "activity": "길거리 음식 체험", "when": "오후", "est_cost_krw": 5000},
  #   {"poi_id": 2, "activity": "등산", "when": "오전", "est_cost_krw": 0},
  #   {"poi_id": 2, "activity": "호수 카약", "when": "오후", "est_cost_krw": 20000},
  # ]

  # if isinstance(acts_df, list):
  #   acts_df = pd.DataFrame(acts_df)

  # 전달 데이터를 아이디 별로 묶어준다.
  act_map = acts_df.groupby("poi_id").apply(lambda df: df.to_dict('records')).to_dict()

  # ranked_pois에서 city_hint라는 특정 도시를 우선순위로 가산점을 준다.
  # city_hint가 있으면 해당 도시에 속한 poi들이 우선순위로 올라간다.
  # test/cand.py 참조
  # 추천 알고리즘이 특정 도시를 우선적으로 고려하도록 가산점을 준다.
  cand = ranked_pois.copy()
  if city_hint and city_hint in set(cand["city"]):
    # city_boost 컬럼 생성: True - 1, False = 0
    # 즉, city_hint에 해당하는 내용이 있으면 1로 우선 순위를 둠
    cand['city_boost'] = (cand["city"] == city_hint).astype(int)
    cand = cand.sort_values(['city_boost', 'score'], ascending=False)
  
  # 코스 생성
  # 예시)
  # 1일차: 서울 후보(가산점 + 높은 score) 위주로 avg_hours 합이 6시간 안에서 담김
  # 2일차: 남은 후보 중 계절/예산에 맞는 곳 채움(서울(지정한 지역)이 부족하면 다른 도시도 들어갈 수 있음)
  # 각 아이템에 최대 2개의 activity 추천

  # 알고리즘
  # 후보지 목록에서 다음 20개 후보지 탐색
  # i를 20개씩 묶음으로 탐색하는 것을 '그리디-윈도우' 탐색이라 한다.
  # 그리디 알고리즘 참조: https://velog.io/@kyunghwan1207/%EA%B7%B8%EB%A6%AC%EB%94%94-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98Greedy-Algorithm-%ED%83%90%EC%9A%95%EB%B2%95
  # 그리디 윈도우 참조: https://wikidocs.net/206308

