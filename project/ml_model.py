# 라이브러리
import seaborn as sns 
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import time
import joblib
import json
from datetime import datetime
# 모델 라이브러리
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# 데이터 가져오기
# 훈련 데이터
train = pd.read_csv("train.csv")
train.head(1)
# 테스트 데이터
test = pd.read_csv("test.csv")
test.head(1)

# 수치형 / 범주형 컬럼 나누기
numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
category_features = ['Sex']

# 독립변수, Target 설정
X = train[numeric_features + category_features]
X
y = train['Calories']
y


# 학습 / 검증 데이터 분할 함수
num_bins = 20
y_binned = pd.cut(y, bins=num_bins, labels=False)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_binned)

# Target 데이터 로그변환 : RMSLE 평가지표를 따라가기 위함
y_train = np.log1p(y_train)
y_train
y_val = np.log1p(y_val)
y_val

# 데이터 가공
# one-hot encoding, scaler
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_category = encoder.fit_transform(X_train[category_features])
X_val_category = encoder.transform(X_val[category_features])

scaler = StandardScaler()
X_train_numeric = scaler.fit_transform(X_train[numeric_features])
X_val_numeric = scaler.transform(X_val[numeric_features])

# 가공한 데이터 합치기 (numeric+category)
X_train_combined = pd.concat([
        pd.DataFrame(X_train_category, columns=['sex1','sex2']),
        pd.DataFrame(X_train_numeric, columns=numeric_features)
    ], axis=1)
X_val_combined = pd.concat([
        pd.DataFrame(X_val_category, columns=['sex1','sex2']),
        pd.DataFrame(X_val_numeric, columns=numeric_features)
    ], axis=1)


# 모델 생성 / 학습 / 성능비교
models = {
    'XGBoost' : XGBRegressor(random_state=42, n_jobs=-1),
    'RandomForest' : RandomForestRegressor(random_state=42, n_jobs=-1),
    'LightGBM' : LGBMRegressor(random_state=42, n_jobs=-1),
    'CatBoost' : CatBoostRegressor(random_seed=42,
                    verbose=100,
                    early_stopping_rounds=50,
                    loss_function='RMSE'
                )
}

# 각 모델별 하이퍼파라미터 후보군 정의
param_grid = {
    'XGBoost': [
        {'n_estimators':100, 'learning_rate': 0.05, 'max_depth': 4},
        {'n_estimators':200, 'learning_rate': 0.01, 'max_depth': 6}
    ],
    'RandomForest': [
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 200, 'max_depth': 10}
    ],
    'LightGBM': [
        {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': -1},
        {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 6}
    ],
    'CatBoost': [
        {'iterations': 500, 'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3, 'subsample': 0.8, 'random_strength': 5},
        {'iterations': 1000, 'learning_rate': 0.1, 'depth': 8, 'l2_leaf_reg': 10, 'subsample': 1.0, 'random_strength': 10}
    ]
}

# 교차 검증을 통한 최적 모델 선택
best_score = 0 # 최고 성능 점수
best_model_name = None
best_model = None


# 각 모델과 하이퍼 파라미터 조합에 대해 교차검증 수행
for model_name, model in models.items():
    print(f"\n--- Testing {model_name} ---")
    start_time = time.time()
    for params in param_grid[model_name]:
        model.set_params(**params)
        cv_scores = cross_val_score(model, X_train_combined, y_train, cv=5, scoring='r2')
        mean_cv = np.mean(cv_scores)  # 평균 교차검증 점수
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Params: {params}, CV R2 Score: {mean_cv:.4f}, Training Time: {elapsed_time:.2f} seconds")
        
        # 최고 성능 모델 업데이트
        if mean_cv > best_score:
            best_score = mean_cv
            best_model_name = model_name
            best_model = model.set_params(**params)


# 최종 선택된 모델로 테스트셋 평가
best_model.fit(X_train_combined, y_train)  # 최적 모델 학습
y_pred = best_model.predict(X_val_combined)  # 테스트셋 예측
test_r2 = r2_score(y_val, y_pred)  # 테스트셋 정확도 계산


# 최고 성능 모델 저장
model_info = {
    'model':best_model,
    'scaler':scaler,
    'encoder':encoder
}
joblib.dump(model_info, 'model.pkl')
# with open('model_info.json', 'w') as f:
#     json.dump(model_info, f)

# 최종 결과 출력
print(f"\nBest Model: {best_model_name}")
print(f"Best CV Score: {best_score:.4f}")
print(f"Test Set R2: {test_r2:.4f}")
print("\nModel has been saved.")