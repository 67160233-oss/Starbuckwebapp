import streamlit as st
import pandas as pd
import joblib

# --- Import สิ่งที่จำเป็นสำหรับแกะไฟล์โมเดล ---
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor # จำเป็นสำหรับโมเดลของคุณ

# 1. โหลดโมเดล (ใช้ Cache เพื่อให้โหลดแค่ครั้งเดียว)
@st.cache_resource
def load_model():
    return joblib.load("starbucks_model.pkl")

model = load_model()

# 2. หัวข้อเว็บ
st.title("☕ Starbucks Spend Predictor")
st.write("ระบบพยากรณ์ยอดใช้จ่ายของลูกค้า")

# 3. ส่วนรับข้อมูลจากผู้ใช้ (เรียงลงมาตรงๆ ไม่แบ่งคอลัมน์)
cart_size = st.number_input("1. จำนวนรายการ (Cart Size)", min_value=1, value=1)
num_customizations = st.number_input("2. การปรับแต่งเครื่องดื่ม", min_value=0, value=0)

order_channel = st.selectbox("3. ช่องทางการสั่งซื้อ", ['Drive-Thru', 'Mobile App', 'In-Store', 'Kiosk'])
drink_category = st.selectbox("4. ประเภทเครื่องดื่ม", ['Coffee', 'Tea', 'Frappuccino', 'Refresher', 'Bakery', 'Other'])
store_location_type = st.selectbox("5. ประเภททำเลร้าน", ['Urban', 'Suburban', 'Rural'])

is_rewards_member = st.checkbox("เป็นสมาชิก Rewards")
has_food_item = st.checkbox("มีการสั่งอาหารร่วมด้วย")
order_ahead = st.checkbox("สั่งล่วงหน้า (Order Ahead)")

# 4. ปุ่มกดเพื่อทำนาย
if st.button("ทำนายยอดใช้จ่าย"):
    
    # สร้างตารางข้อมูลให้ตรงกับที่โมเดลต้องการ
    input_data = pd.DataFrame([{
        'order_channel': order_channel,
        'store_location_type': store_location_type,
        'region': 'Northeast',        # ข้อมูลสมมติที่โมเดลต้องการ
        'customer_age_group': '25-34',# ข้อมูลสมมติที่โมเดลต้องการ
        'customer_gender': 'Other',   # ข้อมูลสมมติที่โมเดลต้องการ
        'is_rewards_member': is_rewards_member,
        'cart_size': cart_size,
        'num_customizations': num_customizations,
        'drink_category': drink_category,
        'has_food_item': has_food_item,
        'order_ahead': order_ahead
    }])

    # สั่งให้โมเดลทำนาย
    prediction = model.predict(input_data)
    
    # แสดงผลลัพธ์แบบเรียบง่าย
    st.success(f"💰 ยอดใช้จ่ายที่คาดการณ์คือ: ${prediction[0]:.2f}")
