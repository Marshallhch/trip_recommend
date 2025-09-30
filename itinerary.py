from datetime import datetime, timedelta # timedelta: 특정 시점부터의 시간 간격
import pandas as pd
from scoring import season_match, budget_ok

# --- 테스트용 POI 후보(ranked_pois) --------------------------------
ranked_pois = pd.DataFrame([
    {"poi_id": 1, "city": "서울", "spot_name": "경복궁",    "tags": "문화|유적",     "avg_hours": 2, "season_suitability": "봄|가을", "budget_hint": "저", "score": 9.2, "lat": 37.5796, "lon": 126.9770},
    {"poi_id": 2, "city": "서울", "spot_name": "남산타워",  "tags": "전망|야경",     "avg_hours": 2, "season_suitability": "사계절",   "budget_hint": "중", "score": 8.8, "lat": 37.5512, "lon": 126.9882},
    {"poi_id": 3, "city": "부산", "spot_name": "해운대",    "tags": "바다|휴식",     "avg_hours": 3, "season_suitability": "여름|가을", "budget_hint": "저", "score": 8.6, "lat": 35.1581, "lon": 129.1604},
    {"poi_id": 4, "city": "부산", "spot_name": "광안대교",  "tags": "야경|드라이브", "avg_hours": 2, "season_suitability": "사계절",   "budget_hint": "저", "score": 8.3, "lat": 35.1531, "lon": 129.1186},
    {"poi_id": 5, "city": "제주", "spot_name": "성산일출봉","tags": "자연|등산",     "avg_hours": 3, "season_suitability": "봄|가을", "budget_hint": "중", "score": 9.0, "lat": 33.4591, "lon": 126.9425},
    {"poi_id": 6, "city": "제주", "spot_name": "협재해수욕장","tags":"바다|휴식",    "avg_hours": 2, "season_suitability": "여름",     "budget_hint": "저", "score": 7.9, "lat": 33.3940, "lon": 126.2394},
])

# --- 테스트용 액티비티(acts_df) -----------------------------------
acts_df = pd.DataFrame([
    {"poi_id": 1, "activity": "한복 체험",   "when": "오전", "est_cost_krw": 15000},
    {"poi_id": 1, "activity": "수문장 교대", "when": "정오", "est_cost_krw": 0},
    {"poi_id": 2, "activity": "전망대 입장", "when": "오후", "est_cost_krw": 16000},
    {"poi_id": 2, "activity": "케이블카",    "when": "오후", "est_cost_krw": 14000},
    {"poi_id": 3, "activity": "모래사장 산책","when": "오전", "est_cost_krw": 0},
    {"poi_id": 4, "activity": "야경 드라이브","when": "야간", "est_cost_krw": 0},
    {"poi_id": 5, "activity": "일출 등반",   "when": "이른새벽","est_cost_krw": 0},
])

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
  act_map = acts_df.groupby("poi_id").apply(lambda df: df.to_dict("records")).to_dict()

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
  # i를 20개씩 묶음으로 탐색하는 것을 '그리디-슬라이딩-윈도우' 탐색이라 한다.
  # 그리디 알고리즘 참조: https://velog.io/@kyunghwan1207/%EA%B7%B8%EB%A6%AC%EB%94%94-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98Greedy-Algorithm-%ED%83%90%EC%9A%95%EB%B2%95
  # 그리디 윈도우 참조: https://wikidocs.net/206308

  d = 0 # 몇칠째 일정인지 초기화(0부터 시작)
  i = 0 # 후보 목록에서 현재 보는 창의 시작 인덱스 초기화(20개씩 증가)
  while d < days and i < len(cand): # 일정이 남아있고 후보지 목록에 남은 위치가 있다면 반복
    day_items, remaining = [], per_day_hours
    used_tags = set() # 이미 사용한 태그

    # 상위 20개 후보지 단위로 체크하여 row에 저장하는 반복문
    for _, row in cand.iloc[i:i+20].iterrows():
      if row['poi_id'] in used_pois: # 이미 방문한 위치면 스킵
        continue
      if not season_match(row["season_suitability"], season): # 계절이 맞지 않으면 스캠
        continue
      if not budget_ok(row["budget_hint"], budget): # 예산이 맞지 않으면 스킵
        continue

      tags = set(str(row["tags"]).split("|")) # 태그 목록을 | 기준으로 분리하여 저장
      if len(tags & used_tags) > 0 and remaining <= per_day_hours//2: # 남은 시간이 하루의 절반보다 작고 이미 사용한 태그와 중복되면 스킵
        continue

      hours = int(row.get("avg_hours", 2)) # 평균 소요 시간, 기본값은 2시간
      if hours <= remaining: # 여행 시간이 남았을 때
        a_list = act_map.get(row["poi_id"], [])[:2] # 활동 목록 중 첫 두 개만 추천
        day_items.append({
          "poi_id": row["poi_id"],
          "city": row["city"],
          "spot_name": row["spot_name"],
          "tags": row["tags"],
          "hours": hours,
          "lat": row["lat"],
          "lon": row["lon"],
          "activities": [
            {
              "activity": a["activity"],
              "when": a["when"],
              "est_cost_krw": a["est_cost_krw"]
              } for a in a_list
          ],
        })
        used_pois.add(row["poi_id"]) # 이미 사용한 위치 아이디 목록에 추가
        used_tags |= tags # 이미 사용한 태그 목록에 추가
        remaining -= hours # 남은 시간 한시간씩 감속
      
      if remaining <= 0:
        break

    if day_items: # 코스가 완성되면
      plan.append({"date": str(start_date + timedelta(days=d)), "items": day_items}) # 코스 목록 추가
      d += 1 # 다음 날짜로 이동

    i += 20
    if i >= len(cand) and d < days: # 후보지 목록의 개수보다 크거나 같고, 일정이 남아 있으면:
      i = 0 # 후보지 재탐색
      per_day_hours = max(4, per_day_hours - 1) # 하루 시간 줄인다. 최소 4시간

  return plan

if __name__ == "__main__":
  result = generate_itinerary(ranked_pois=ranked_pois, acts_df=acts_df, days=2, season="봄", budget="저", city_hint="서울", per_day_hours=6)

  from pprint import pprint
  pprint(result)


