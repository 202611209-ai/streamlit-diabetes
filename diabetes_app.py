import streamlit as st
import pandas as pd
import numpy as np
import os

# joblib 자동 설치 및 로드
try:
    import joblib
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    import joblib

# 1. 페이지 레이아웃 및 쌈뽕한 디자인
st.set_page_config(
    page_title="당뇨병 위험도 예측 AI",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 25px;
    }
    .result-box {
        padding: 25px;
        border-radius: 15px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# 2. 📂 [중요] 모델과 스케일러를 joblib으로 안전하게 로드
@st.cache_resource
def load_ml_components():
    model_path = "당뇨예측모델 (1).pkl"
    scaler_path = "scaler.pkl"
    
    # 파일이 비어있거나 없는지 사전 검사 (EOFError 방지)
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        st.error(f"❌ '{model_path}' 파일이 없거나 크기가 0바이트(깨짐)입니다! 코랩에서 다시 다운로드 받아주세요.")
        st.stop()
    if not os.path.exists(scaler_path) or os.path.getsize(scaler_path) == 0:
        st.error(f"❌ '{scaler_path}' 파일이 없거나 크기가 0바이트(깨짐)입니다! 코랩에서 함께 다운로드 받아주세요.")
        st.stop()
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    log_model_eng, scaler = load_ml_components()
except Exception as e:
    st.error(f"❌ 파일 로드 중 에러 발생 (코랩에서 새로 추출하는 것을 권장합니다): {e}")
    st.stop()

# 3. 타이틀 표시
st.markdown('<div class="main-title">🩺 쌈뽕한 AI 당뇨병 발병 예측 시스템</div>', unsafe_allow_html=True)
st.write("사이드바에 값을 입력하면 실시간으로 Scaler 변환 및 파생변수 생성이 진행되어 정밀하게 예측합니다.")
st.markdown("---")

# 4. 👤 사이드바 입력 폼 (input 함수 대체)
st.sidebar.header("📊 건강 지표 데이터 입력")
preg = st.sidebar.number_input("🤰 임신 횟수 입력:", min_value=0, max_value=20, value=2, step=1)
glucose = st.sidebar.number_input("🩸 혈당 입력:", min_value=0.0, max_value=300.0, value=130.0, step=1.0)
bp = st.sidebar.number_input("💓 혈압 입력:", min_value=0.0, max_value=200.0, value=130.0, step=1.0)
skin = st.sidebar.number_input("📏 피부두께 입력:", min_value=0.0, max_value=100.0, value=12.0, step=1.0)
insulin = st.sidebar.number_input("💉 인슐린 입력:", min_value=0.0, max_value=900.0, value=26.0, step=1.0)
bmi = st.sidebar.number_input("🏃‍♂️ 체질량지수(BMI) 입력:", min_value=0.0, max_value=70.0, value=26.0, step=0.1)
dpf = st.sidebar.number_input("🧬 당뇨병 혈통 기능 입력:", min_value=0.0, max_value=3.0, value=0.8, step=0.01)
age = st.sidebar.number_input("🎂 나이 입력:", min_value=1, max_value=120, value=45, step=1)

# 5. 메인 화면 데이터 대시보드 요약
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("혈당 수치", f"{glucose} mg/dL")
    st.metric("임신 횟수", f"{preg} 회")
with col2:
    st.metric("혈압 수치", f"{bp} mmHg")
    st.metric("나이", f"{age} 세")
with col3:
    st.metric("체질량지수(BMI)", f"{bmi}")
    st.metric("피부 두께", f"{skin} mm")
with col4:
    st.metric("당뇨 혈통 지수", f"{dpf}")
    st.metric("인슐린 수치", f"{insulin} mIU/L")

st.markdown("---")

# 6. 🧪 [데이터 전처리] DataFrame 생성 및 6개 파생 변수 계산
input_data = pd.DataFrame(
    [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
    columns=['임신횟수', '혈당', '혈압', '피부두께', '인슐린', '체질량지수', '당뇨병 혈통 기능', '나이']
)

input_data['인슐린저항성지표'] = input_data['혈당'] * input_data['인슐린']
input_data['비만혈압지수'] = input_data['체질량지수'] * input_data['혈압']
input_data['신체지방지표'] = input_data['피부두께'] + input_data['체질량지수']
input_data['연령별당위험'] = input_data['나이'] * input_data['혈당']
input_data['임신위험도'] = input_data['임신횟수'] * input_data['나이']
input_data['유전당위험'] = input_data['당뇨병 혈통 기능'] * input_data['혈당']

# 7. 🤖 [실시간 예측] 불러온 스케일러로 transform 후 모델 예측
st.subheader("🧠 AI 실시간 분석 및 진단 결과")

try:
    # 🌟 스케일러로 변환 작업 처리 후 모델에 전달!
    input_scaled = scaler.transform(input_data)
    
    predicted = log_model_eng.predict(input_scaled)
    prob = log_model_eng.predict_proba(input_scaled)
    diabetes_risk_percent = prob[0][1] * 100

    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        if predicted[0] == 0:
            st.markdown(
                f'<div class="result-box" style="background-color: #E3F2FD; color: #0D47A1; border: 2px solid #2196F3;">'
                f'🎉 분석 결과: 정상 (0)<br>'
                f'<span style="font-size: 15px; font-weight: normal;">당뇨병 위험 진단 결과가 매우 낮고 안전한 상태입니다!</span>'
                f'</div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box" style="background-color: #FFEBEE; color: #B71C1C; border: 2px solid #F44336;">'
                f'⚠️ 분석 결과: 당뇨 위험군 검출 (1)<br>'
                f'<span style="font-size: 15px; font-weight: normal;">당뇨 발병 확률이 높은 위험군에 속하므로 예방이 시급합니다!</span>'
                f'</div>', 
                unsafe_allow_html=True
            )
            
    with res_col2:
        st.write(f"📊 **당뇨병 발생 확률:** {diabetes_risk_percent:.1f}%")
        st.progress(int(diabetes_risk_percent))
        
        if diabetes_risk_percent < 30:
            st.success("🟢 안심 단계: 현재의 건강 상태를 잘 유지해 주세요.")
        elif diabetes_risk_percent < 70:
            st.warning("🟡 주의 단계: 꾸준한 식단 관리 및 운동이 필요합니다.")
        else:
            st.error("🔴 고위험 단계: 인슐린 조절 및 정밀 의학 진단을 추천합니다.")

except Exception as prediction_error:
    st.error(f"⚠️ 예측 연동 에러 발생: {prediction_error}")