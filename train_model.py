import joblib
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer # tuts/mlb.py 참조
from sklearn.pipeline import Pipeline # mlb, vectorizer를 연결해줄 모듈
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB # tuts/cnb.py
from sklearn.multiclass import OneVsRestClassifier # tuts/ovr.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import INTEREST_THRESHOLD, DATA_DIR, MODELS_DIR

df = pd.read_csv(DATA_DIR / 'users_sample.csv')

def user_text(df: pd.DataFrame) -> pd.Series:
  """
  사용자 프로필을 텍스트로 합쳐 TF-IDF 입력으로 사용
  Args: 
    df: 사용자 데이터 프레임
  Returns:
    pd.Series: 사용자 프로필 중 지정 컬럼을 필터링한 텍스트
  """
  cols = ['party', 'season', 'style', 'budget_level', 'origin_city']
  for c in cols:
    if c not in df.columns:
      df[c] = ""
  return (df['party'] + " " + df['season'] + " " + df['style'] + " " + df['budget_level'] + " " + df['origin_city']).fillna("")

# print(user_text(df))

def train_interest_classifier(df: pd.DataFrame):
  """
  멀티라벨 관심사 분류기 학습 및 저장
  Args:
    df: 사용자 데이터 프레임
  Returns:
    clf: 관심사 분류기
    mlb: 멀티라벨 이진화
  """
  mlb = MultiLabelBinarizer()
  x = user_text(df)
  y = mlb.fit_transform(df["interests"].str.split("|"))

  clf = Pipeline([
    ('tfidf', TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")), # 한글 패턴 토큰화
    ("ovr", OneVsRestClassifier(ComplementNB()))
  ])

  clf.fit(x, y)

  # 생성 모델 저장
  MODELS_DIR.mkdir(parents=True, exist_ok=True)
  joblib.dump(clf, MODELS_DIR / "interests_clf.joblib")
  joblib.dump(mlb, MODELS_DIR / "mlb.joblib")

  return clf, mlb

# train_interest_classifier(df)

# 훈련 모델 로드 함수
def load_interest_classifier():
  """
  저장된 분류기/라벨 바이너리 로드
  Returns:
    clf: 관심사 분류기
    mlb: 멀티라벨 이진화
  """
  clf = joblib.load(MODELS_DIR / 'interests_clf.joblib')
  mlb = joblib.load(MODELS_DIR / 'mlb.joblib')
  return clf, mlb

# 관심사 추론(태그 리스트 또는 (태그, 점수) 튜플 반환)
def infer_interests(profile_row: dict, clf, mlb, top_k=5, return_scores=False):
  """
  사용자 프로필로부터 관심사 태그 추론
  Args:
    profile_row: 사용자 프로필 딕셔너리
    clf: 관심사 분류기
    mlb: 멀티라벨 이진화
    top_k: 반환할 상위 관심사 개수
    return_scores: True면 점수화 함께 반환
  Returns:
    list 또는 list of tuples: 관심사 태그 리스트 또는 [(태그, 점수)] 형식의 퓨틀 리스트
  """
  df = pd.DataFrame([profile_row])
  probs = clf.predict_proba(user_text(df))[0] # 첫 번째 사용자 확률만 가져옴

  # 레이블과 확률 매핑
  labels = mlb.classes_
  # [('foodie', 0.9), ("nature", '0.8')]
  label_scores = list(zip(labels, probs))

  # 확률 순서로 정렬
  label_scores.sort(key=lambda x: x[1], reverse=True)

  if return_scores:
    return label_scores[:top_k]
  else:
    # 임계값을 넘는 것들 중 상위 top_k개의 레이블 반환
    high_confidence = [(label, score) for label, score in label_scores if score > INTEREST_THRESHOLD]

    if len(high_confidence) > top_k:
      # 임계값을 넘는 것들 중 상위 top_k개의 레이블 반환
      return [label for label, _ in high_confidence[:top_k]]
    else:
      # 임계값을 넘는 것이 적다면 상위 top_k개 반환
      return [label for label, _ in label_scores[:top_k]]

# 관심 태그 추론(태그, 점수, 추천여부) 튜플 반환
def infer_interests_with_score(profile_row: dict, clf, mlb, top_k=10):
  """
  점수와 함께 관심사 태그 추론(상세 분석)
  Args:
    profile_row: 사용자 프로필 딕셔너리
    clf: 관심사 분류기
    mlb: 멀티라벨 이진화
    top_k: 반환할 상위 관심사 개수
    return_scores: True면 점수화 함께 반환
  Returns:
    list 또는 list of tuples: 관심사 태그 리스트 또는 [(태그, 점수)] 형식의 퓨틀 리스트
  """

  df = pd.DataFrame([profile_row])
  probs = clf.predict_proba(user_text(df))[0] # 첫 번째 사용자 확률만 가져옴

  # 레이블과 확률 매핑
  labels = mlb.classes_
  # [('foodie', 0.9), ("nature", '0.8')]
  label_scores = []

  for label, score in zip(labels, probs): # 레이블과 확률을 매칭
    is_recommended = score > INTEREST_THRESHOLD # 임계값 초과 시 추천
    label_scores.append((label, score, is_recommended))

  # 확률 순으로 정렬
  label_scores.sort(key=lambda x: x[1], reverse=True)

  return label_scores