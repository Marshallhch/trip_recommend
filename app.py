import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.multiclass import OneVsRestClassifier
from datetime import datetime, timedelta
import itertools, random
import math

st.set_page_config(page_title="여행 코스 생성기", layout="wide")

@st.cache_data # 데이터 캐시 사용
def load_data():
  users = pd.read_csv("data/users_sample.csv")
  pois = pd.read_csv("data/pois_korea.csv")
  acts = pd.read_csv("data/activities.csv")
  return users, pois, acts

# 하버사인 함수
def haversine_distance(lat1, lon1, lat2, lon2):
  """
  두 지점 간의 거리를 킬로미터 단위로 계산 (하버사인 공식)
  """
  R = 6371  # 지구의 반지름 (km)
  
  # 위도와 경도를 라디안으로 변환
  lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
  
  # 위도와 경도의 차이
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  
  # 하버사인 공식
  a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
  c = 2 * math.asin(math.sqrt(a))
  
  return R * c

# 기준 도시 좌표 매핑
# 기준 도시의 좌표 매핑
CITY_COORDS = {
  "서울": (37.5665, 126.9780),
  "부산": (35.1796, 129.0756),
  "제주": (33.4996, 126.5312),
  "경주": (35.8562, 129.2247)
}

def get_city_distance(base_city, target_lat, target_lon):
  """
  기준 도시로부터 대상 지점까지의 거리를 계산
  """
  if base_city not in CITY_COORDS:
      # 기준 도시가 매핑되지 않은 경우, POI 데이터에서 해당 도시의 평균 좌표 사용
      return 50  # 기본값으로 50km 설정
  
  base_lat, base_lon = CITY_COORDS[base_city]
  return haversine_distance(base_lat, base_lon, target_lat, target_lon)

users, pois, acts = load_data()

st.title("여행 코스 생성기")
st.caption('간단한 선호도 분류(나이브 베이즈) + 규칙 기반 코스')

with st.sidebar:
  st.header('여행 설정')
  days = st.slider('여행 일수', 1, 3, 5, step=1)
  party = st.selectbox('일행', ['solo', 'couple', 'friends', 'family'])
  season = st.selectbox('계절', ['봄', '여름', '가을', '겨울'])
  style = st.selectbox('여행 스타일', ['relax', 'active', 'balanced'])
  budget = st.selectbox('예산', ['하', '중', '상'])
  base_city = st.selectbox('출발 도시', sorted(pois["city"].unique().tolist()+["서울", "부산", "제주", "경주"]))

  # 키워드 선택
  all_tags = sorted(set(t for row in pois["tags"] for t in str(row).split("|")))
  interests = st.multiselect('관심사(복수선택)', all_tags, default=["foodie", "culture"])

#==================== model training base interests filtering ======================# 
def user_text(df):
  return (df['party'] + " " + df['season'] + " " + df['style'] + " " + df['budget_level'] + " " + df['origin_city']).fillna("")

if users.empty:
  st.stop()

mlb = MultiLabelBinarizer()
x = user_text(users)
y = mlb.fit_transform(users["interests"].str.split("|"))

clf = Pipeline([
  ('tfidf', TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")), # 한글 패턴 토큰화
  ("ovr", OneVsRestClassifier(ComplementNB()))
])

clf.fit(x, y)

current_profile = pd.DataFrame([
  {
    "party": party,
    "season": season,
    "style": style,
    "budget_level": budget,
    "origin_city": base_city
  }
])

pred_probs = clf.predict_proba(user_text(current_profile))
pred_interest = mlb.inverse_transform((pred_probs > 0.35).astype(int))

if pred_interest:
  inferred = sorted(pred_interest[0])
else:
  inferred = []

st.subheader('예상 관심사 태그')
st.write(", ".join(inferred) if inferred else "유의미한 태그를 찾지 못했습니다. (기본값 사용)")

#==================== rule(점수 생성) base filtering ======================# 
def season_match(row_season, season: str) -> bool:
  if pd.isna(row_season): 
    return True
  sset = set(str(row_season).split("|"))
  return (season in sset) or ('사계절' in sset)

def budget_ok(row_hint: str, budget: str) -> bool:
  order = {'하': 0, '중': 1, '상': 2}
  return order.get(row_hint, 1) <= order.get(budget, 1)

def score_poi(row, inferred_tags, user_interests, base_city):
 
  tags = set(str(row["tags"]).split("|"))
  overlap_inferred = len(tags & set(inferred_tags)) # 추론 태그하고 명소 태그하고 중복되는 개수
  overlap_user = len(tags & set(user_interests)) # 사용자 관심 태그와 명소 태그 중복 개수
  s_bonus = 1 if season_match(row['season_suitability'], season) else 0
  b_bonus = 1 if budget_ok(row["budget_hint"], budget) else 0

  # 거리 기반 가중치 계산 (강화된 필터링)
  distance = get_city_distance(base_city, row["lat"], row["lon"])
  # 거리 제한을 더욱 강화
  if distance > 200:
      distance_score = -20  # 200km 이상은 매우 큰 마이너스 점수
  elif distance > 150:
      distance_score = -10  # 150-200km는 큰 마이너스 점수
  elif distance > 100:
      distance_score = 0    # 100-150km는 중립
  elif distance > 50:
      distance_score = 3    # 50-100km는 보통 점수
  else:
      distance_score = 8    # 50km 이내는 매우 높은 점수
  return (2 * overlap_user) + (1.5 * overlap_inferred) + s_bonus + b_bonus + distance_score

user_interests = interests if interests else ["foodie"]
pois = pois.copy() 
pois["score"] = pois.apply(lambda r: score_poi(r, inferred, user_interests, base_city), axis=1)
ranked = pois.sort_values(["score", "city"], ascending=False).reset_index(drop=True) 

st.subheader("추천 후보 명소(가중치 기준)")

# 거리 정보
display_ranked = ranked.copy()
display_ranked["distance_km"] = display_ranked.apply(lambda r: round(get_city_distance(base_city, r['lat'], r["lon"]), 1), axis=1)
st.dataframe(display_ranked[['city','spot_name','tags','season_suitability','budget_hint','distance_km', 'score']].head(7))

#==================== itinerary generator ======================# 
def generate_itinerary(ranked_pois, acts, days, city_hint=None):
    """
    지역 연속성을 고려한 일정 생성
    - 첫날은 출발 도시 기준으로 선택
    - 이후 날짜는 전날 마지막 위치에서 가까운 곳 우선
    - 일일 최대 이동거리 150km 제한
    """
    plan = []
    start_date = datetime.now().date()
    per_day_hours = 6
    used = set()
    max_daily_distance = 150  # 일일 최대 이동거리 (km)
    
    # 현재 위치 (초기: 출발 도시)
    current_location = None
    if city_hint and city_hint in CITY_COORDS:
        current_location = CITY_COORDS[city_hint]
    
    # Pre-index activities by poi
    act_map = acts.groupby("poi_id").apply(lambda df: df.to_dict("records")).to_dict()

    for day_num in range(days):
        day_items = []
        remaining_hours = per_day_hours
        used_tags_today = set()
        
        # 현재 위치에서 가까운 POI들 선택
        available_pois = ranked_pois[~ranked_pois["poi_id"].isin(used)].copy()
        
        if current_location:
            # 현재 위치에서의 거리 계산
            available_pois["distance_from_current"] = available_pois.apply(
                lambda r: haversine_distance(current_location[0], current_location[1], r["lat"], r["lon"]), axis=1
            )
            # 일일 이동거리 제한 적용
            available_pois = available_pois[available_pois["distance_from_current"] <= max_daily_distance]
            
            # 거리 기반 재정렬 (가까운 곳 우선 + 기존 점수 고려)
            available_pois["daily_score"] = available_pois["score"] - (available_pois["distance_from_current"] / 10)
            available_pois = available_pois.sort_values("daily_score", ascending=False)
        else:
            # 첫날이고 출발 도시가 명확하지 않은 경우 기본 순서 사용
            available_pois = available_pois.sort_values("score", ascending=False)
        
        # 하루 일정 구성
        for _, row in available_pois.head(30).iterrows():  # 상위 30개 중에서 선택
            if remaining_hours <= 0:
                break
                
            # 계절, 예산 체크
            if not season_match(row["season_suitability"], season):
                continue
            if not budget_ok(row["budget_hint"], budget):
                continue
                
            # 태그 다양성 체크 (하루 내에서)
            tags = set(str(row["tags"]).split("|"))
            if len(tags & used_tags_today) > 1 and remaining_hours <= per_day_hours // 2:
                continue  # 너무 비슷한 태그는 피하기
                
            avg_h = int(row.get("avg_hours", 2))
            if avg_h <= remaining_hours:
                # 활동 정보 추가
                a_list = act_map.get(row["poi_id"], [])
                add_acts = []
                for a in a_list[:2]:
                    add_acts.append({
                        "activity": a["activity"], 
                        "when": a["when"], 
                        "est_cost_krw": a["est_cost_krw"]
                    })
                
                day_items.append({
                    "poi_id": row["poi_id"],
                    "city": row["city"],
                    "spot_name": row["spot_name"],
                    "tags": row["tags"],
                    "hours": avg_h,
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "activities": add_acts
                })
                
                used.add(row["poi_id"])
                used_tags_today |= tags
                remaining_hours -= avg_h
                
                # 현재 위치 업데이트 (마지막 방문지로)
                current_location = (row["lat"], row["lon"])
        
        # 하루 일정이 구성되었으면 추가
        if day_items:
            plan.append({
                "date": str(start_date + timedelta(days=day_num)), 
                "items": day_items
            })
        else:
            # 일정을 구성할 수 없으면 제약 완화
            max_daily_distance += 50  # 이동거리 제한 완화
            if max_daily_distance > 300:  # 최대 300km까지만 허용
                break
    
    return plan

itinerary = generate_itinerary(ranked, acts, days=days, city_hint=base_city)

st.subheader('생성된 일정')
for day in itinerary:
   st.markdown(f'Day: {day["date"]}')
   for item in day["items"]:
      st.markdown(f'{item["city"]} / {item["spot_name"]} / {item["hours"]}시간 - _{item["tags"]}_')
      for a in item["activities"]:
         st.markdown(f"     - {a['when']} / {a['activity']} / 예상 비용 약 {a['est_cost_krw']}원")