# 하나의 데이터에 여러 개의 라벨이 붙을 수 있을 때, 이를 머신러닝 모델이 처리할 수 있는 숫자 벡터 형태로 변환
# 하나의 데이터에 여러 라벨이 있는 경우. 예) 영화 -> 액션, 코메디 뉴스 -> 경제, 정치
# 이러한 경우를 멀티라벨 분류라 한다

from sklearn.preprocessing import MultiLabelBinarizer

data = [
  ["foodie", "nature"],
  ["culture"],
  ["foodie", "culture"]
]

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data)

print('classes: ', mlb.classes_)
print('result mlb: ', y)

# ['culture', 'foodie', 'nature']
# ['foodie', 'nature'] -> [0, 1, 1] -> 위 클래스 목록 기준으로 culture는 0(없음), foodie는 1(있음), nature는 1(있음)
# ['culture'] -> [1, 0, 0] -> 위 클래스 목록 기준으로 culture는 1(있음), foodie는 0(없음), nature는 0(없음)