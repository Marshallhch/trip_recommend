import sys
from pathlib import Path

# 루트 디렉토리
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from train_model import load_interest_classifier, infer_interests, infer_interests_with_score

# 다양한 사용자 프로필 테스트
test_profiles = [
  {
    "name": "가족 여행객",
    "profile": {"party": "family", "season": "봄", "style": "relax", "budget_level": "중", "origin_city": "Seoul"}
  },
  {
    "name": "커플 여행객", 
    "profile": {"party": "couple", "season": "가을", "style": "active", "budget_level": "상", "origin_city": "Busan"}
  },
  {
    "name": "솔로 여행객",
    "profile": {"party": "solo", "season": "여름", "style": "balanced", "budget_level": "중", "origin_city": "Seoul"}
  },
  {
    "name": "친구 여행객",
    "profile": {"party": "friends", "season": "겨울", "style": "active", "budget_level": "하", "origin_city": "Daegu"}
  }
]

# 모델 로드
clf, mlb = load_interest_classifier()

for test_case in test_profiles:
  name = test_case["name"]
  profile = test_case["profile"]

  # 상위 5개 추천
  top5_interests = infer_interests(profile, clf, mlb, top_k=5)

  # print(f"추천 관심사: {top5_interests}")

  detailed = infer_interests_with_score(profile, clf, mlb, top_k=8)
  print('상세 점수 분석: ')
  print("=" * 50)
  for i, (interest, score, is_rec) in enumerate(detailed, 1):
    rec_mark = '추천' if is_rec else '비추천'
    # 예시: 1    foodie          0.9000 추천
    # <3: 3자리 왼쪽 정렬, <15: 15자리 왼쪽 정렬, <8: 8자리 왼쪽 정렬, <4: 4자리 왼쪽 정렬
    print(f'{i:<3} {interest:<15} {score:<8.4f} {rec_mark:<4}')