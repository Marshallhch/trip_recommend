# 머신러닝에서 두 개의 클래스가 존재하는 것을 이진 분류(binary classification) 문제라 한다.
# 하지만 클래스가 3개 이상일 경우 다중 분류(multiclass classification) 문제라 한다.
# 다중 분류 문제를 해결 할 때 이진 분류를 여러 개 조합하여 해결하는 방법에는 일대일 분류(One-vs-One) 방법이 있다.

# 클래스가 K개라면 가능한 클래스 쌍마다 이진 분류기를 만든다.
# 그럴 경우 K * (K - 1) / 2 개의 이진 분류기가 필요하다.

# A, B, C, D 클래스가 있을 경우: 4 * (4 - 1) / 2 = 6
# (A, B), (A, C), (A, D), (B, A), (B, C), (B, D), (C, A), (C, B), (C, D), (D, A), (D, B), (D, C)
# ((A, B), (B, A)) -> 중복
# ((A, C), (C, A)) -> 중복
# ((A, D), (D, A)) -> 중복
# ((B, C), (C, B)) -> 중복
# ((B, D), (D, B)) -> 중복
# ((C, D), (D, C)) -> 중복
# 총 6개의 이진 분류기가 필요하다.
# 각각의 이진 쌍 중 가장 많은 선택을 받은 클래스를 최종 분류 결과로 선택한다.

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score # 정확도 분석용

# 데이터 호출
iris = datasets.load_iris()
x = iris.data
y = iris.target

# 데이터 클래스 확인
# print(iris.target_names)

# 데이터 차원 확인
# print(x.shape, y.shape)

# 훈련 데이터 분리: 7:3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# ovo 분류 적용: 서포트 벡터 분류(SVC) 사용
ovo_clf = OneVsOneClassifier(SVC(kernel="linear"))

# 모델 학습
ovo_clf.fit(x_train, y_train)

# 모델 예측
y_pred = ovo_clf.predict(x_test)

# 모델 평가
print("Accuracy: ", accuracy_score(y_test, y_pred))

# 5개의 결과 예측
for i in range(5):
  print(f'실제: {y_test[i]}, 예측: {y_pred[i]}')

# One vs Rest 방식은 다중 분류 문제를 여러 개의 이진 분류 문제로 바꿔서 해결하는 방식이다.
# 클래스가 K개 있으면, 각 클래스마다 1개의 분류기를 만든다.
# 각 분류기는 '선택 클래스 vs 나머지 클래스' 구조로 학습니다.

# a, b, c, d 클래스는 4개의 분류기를 만든다
# a 분류기는 'a vs b, c, d' 구조로 학습
# b 분류기는 'b vs a, c, d' 구조로 학습
# c 분류기는 'c vs a, b, d' 구조로 학습
# d 분류기는 'd vs a, b, c' 구조로 학습

# 이후 모든 분류기가 점수를 평가한다.
# 가장 높은 점수를 받은 클래스가 최종 예측 결과가 된다.