import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- ส่วนสำคัญ: ต้อง Import สิ่งเหล่านี้เพื่อให้ Pipeline โหลดได้ (อ้างอิง Lab 14) ---
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor  # หรือโมเดลที่คุณใช้เทรน

# -------------------------------------------------------------------------

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Starbucks Spend Predictor", page_icon="☕")


# ฟังก์ชันโหลดโมเดล (ใช้แคชเพื่อประสิทธิภาพตาม Lab 14)
@st.cache_resource
def load_model():
    # ตรวจสอบว่าชื่อไฟล์ใน GitHub ตรงกับชื่อนี้
    return joblib.load("starbucks_model.pkl")


with st.spinner("กำลังโหลดโมเดล..."):
    model = load_model()

st.title("☕ Starbucks Spend Predictor")
st.write("ทายยอดใช้จ่ายจากพฤติกรรมการสั่งซื้อ")

# --- ส่วนรับข้อมูลจากผู้ใช้ ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        cart_size = st.number_input("จำนวนรายการ (Cart Size)", min_value=1, value=1)
        num_customizations = st.number_input("การปรับแต่ง", min_value=0, value=0)
        order_channel = st.selectbox("ช่องทาง", ['Drive-Thru', 'Mobile App', 'In-Store', 'Kiosk'])
        drink_category = st.selectbox("ประเภท", ['Coffee', 'Tea', 'Frappuccino', 'Refresher', 'Bakery', 'Other'])

    with col2:
        is_rewards_member = st.checkbox("สมาชิก Rewards")
        has_food_item = st.checkbox("สั่งอาหารด้วย")
        store_location_type = st.selectbox("ทำเล", ['Urban', 'Suburban', 'Rural'])
        order_ahead = st.checkbox("สั่งล่วงหน้า")

    submit = st.form_submit_state = st.form_submit_button("ทำนายผล")

if submit:
    # สร้าง DataFrame ให้เหมือนตอน Train (ลำดับคอลัมน์สำคัญมาก)
    input_data = pd.DataFrame([{
        'order_channel': order_channel,
        'store_location_type': store_location_type,
        'region': 'Northeast',  # ค่า Default
        'customer_age_group': '25-34',  # ค่า Default
        'customer_gender': 'Other',  # ค่า Default
        'is_rewards_member': is_rewards_member,
        'cart_size': cart_size,
        'num_customizations': num_customizations,
        'drink_category': drink_category,
        'has_food_item': has_food_item,
        'order_ahead': order_ahead
    }])

    # ทำนาย
    prediction = model.predict(input_data)

    st.success(f"### ยอดใช้จ่ายโดยประมาณ: ${prediction[0]:.2f}")