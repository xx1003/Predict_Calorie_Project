import pandas as pd
import numpy as np
import joblib
import streamlit as st

st.title("칼로리 소모량 예측기")

# 모델 불러오기
@st.cache_resource
def load_model():
    model_package = joblib.load('model.pkl')
    model = model_package['model']
    encoder = model_package['encoder']
    scaler = model_package['scaler']

    return model, encoder, scaler

model, encoder, scaler = load_model()

def preprocess_input(data, encoder, scaler):
    df = pd.DataFrame([data])
    
    category_features = ['Sex']
    numeric_features = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp']
    # 성별 변환
    cate = encoder.transform(df[category_features])
    # 수치형 변환
    numer = scaler.transform(df[numeric_features])

    df_combined = pd.concat([
        pd.DataFrame(cate, columns=['sex1','sex2']),
        pd.DataFrame(numer, columns=numeric_features)
    ], axis=1)

    return df_combined
    

# 입력 폼 생성
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        height = st.number_input("Height", min_value=0.0, max_value=10000.0, value=160.0)
        weight = st.number_input("Weight", min_value=0.0, max_value=10000.0, value=50.0)

    with col2:
        duration = st.number_input("Duration", min_value=0.0, max_value=10000.0, value=0.0)
        heart_rate = st.number_input("Heart_Rate", min_value=0.0, max_value=300.0, value=100.0)
        body_temp = st.number_input("Body_Temp", min_value=0.0, max_value=100.0, value=35.6)

    submitted = st.form_submit_button("칼로리 소모량 예측하기")

if submitted:
    # 입력 데이터 준비
    input_data = {
        'Sex':sex,
        'Age':age,
        'Height':height,
        'Weight':weight,
        'Duration':duration,
        'Heart_Rate':heart_rate,
        'Body_Temp':body_temp
    }

    # 전처리
    processed_data = preprocess_input(input_data, encoder, scaler)
    prediction = model.predict(processed_data)[0]

    st.markdown("---")
    st.subheader("예측 결과")

    st.metric("칼로리 소모량", f"{prediction}")