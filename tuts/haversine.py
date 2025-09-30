import math

def haversine_distance_verbose(lat1, lon1, lat2, lon2):
  """
  하버사인 공식을 단계별로 보여주며 거리 계산
  
  Args:
      lat1, lon1: 출발지 위도, 경도 (도)
      lat2, lon2: 도착지 위도, 경도 (도)
      show_steps: 계산 과정 출력 여부
  
  Returns:
      거리 (km)
  """
  R = 6371  # 지구의 반지름 (km)

  # 위도와 경도를 라디안으로 변환
  lat1_rad = math.radians(lat1)
  lon1_rad = math.radians(lon1)
  lat2_rad = math.radians(lat2)
  lon2_rad = math.radians(lon2)

  # 위도와 경도 차이 계산
  dlat = lat2_rad - lat1_rad
  dlon = lon2_rad - lon1_rad

  # 하버사인 공식 적용
  # a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
  sin_dlat_half = math.sin(dlat/2)
  sin_dlon_half = math.sin(dlon/2)
  cos_lat1 = math.cos(lat1_rad)
  cos_lat2 = math.cos(lat2_rad)

  a = sin_dlat_half**2 + cos_lat1 * cos_lat2 * sin_dlon_half**2

  # c = 2 × arcsin(√a)
  sqrt_a = math.sqrt(a)
  c = 2 * math.asin(sqrt_a)

  # 최종 거리 R * c
  distance = R * c

  return distance

cities = {
  "서울": (37.5665, 126.9780),
  "부산": (35.1796, 129.0756),
  "인천": (37.4563, 126.7052),
  "수원": (37.2636, 127.0286),
  "제주": (33.4996, 126.5312)
}

seoul_lat, seoul_lon = cities["서울"]
incheon_lat, incheon_lon = cities["제주"]

distance = haversine_distance_verbose(seoul_lat, seoul_lon, incheon_lat, incheon_lon)
print(distance)

