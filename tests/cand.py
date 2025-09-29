import pandas as pd

ranked_pois = pd.DataFrame({
  "poi_id": [1, 2, 3, 4, 5],
  "city": ['서울', '부산', '서울', '제주', '대전'],
  "score": [8.5, 9.0, 7.0, 8.0, 5.0]
})

city_hint = "서울"

cand = ranked_pois.copy()
if city_hint and city_hint in set(cand["city"]):
  # city_boost 컬럼 생성: True - 1, False = 0
  # 즉, city_hint에 해당하는 내용이 있으면 1로 우선 순위를 둠
  cand['city_boost'] = (cand["city"] == city_hint).astype(int)
  cand = cand.sort_values(['city_boost', 'score'], ascending=False)

print(cand)