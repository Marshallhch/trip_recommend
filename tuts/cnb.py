# 나이브 베이즈: 확률 규칙을 이용해 텍스트 같은 데이터를 분류하는 알고리즘이다.
# 예) 메일이 '무료', '담첨' 같은 단어를 많이 포함하면 스펨일 확률이 높다.
# 나이브 베이즈는 계산이 빠르고 적은 데이터만 있어도 잘 작동한다.
# 하지만 클래스 불균형 문제가 자주 발생한다.
# 예를 들어 10,000개의 메일 중 스팸은 500개만 있다면 비율이 너무 차이가 많다.
# 이 경우 소수 클래스인 스팸의 예측 성능이 떨어진다.

# 이를 보완하기 위해 나온 알고리즘에 ComplementNB다.
# 이는 해당 클래스의 데이터만 보는 것이 아니라 나머지 클래스(complement)의 데이터도 함께 고려한다.
# 즉, 클래스마다 해당 클래스가 아닌 데이터들의 통계를 이용해 가중치를 조정한다.

# 나이브 베이즈의 경우 스팸메일 안에서 '무료'가 얼마나 빈번하게 등장하는지만 평가한다.
# 이에 반해 ComplementNB는 스팸이 아닌 정상 메일 안에서 '무료'가 얼마나 적게 나오는지도 평가한다.

# 따라서 클래스 간 데이터 개수가 크게 차이날 때(불균형 데이터) 성능이 안정적이다.
# 주로 TfidfVectorizer와 함께 사용된다.

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB

docs = ["이 제품 정말 좋아요", "무료 당첨 이벤트", "완전 만족합니다", "광고 문구 클릭하세요"]
labels = ["정상", "스팸", "정상", "스팸"]

clf = Pipeline([
  ("tfidf", TfidfVectorizer()),
  ('cnb', ComplementNB())
])

clf.fit(docs, labels)

print(clf.predict(['이 영화 정말 스릴 넘쳐요.']))
print(clf.predict(['쿠폰에 당첨됐습니다. 축하합니다.']))