import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import numpy as np
from datetime import datetime

# Centered, YouTube Content Monetization Modeler
st.image(r"D:\DS_Content_Monetization_Modeler/youtube.png", width=250)
st.markdown(
    """
    <h1 style='text-align: center; color: #9B59B6; font-size: 40px; font-weight: bold;'>
    ▶️ YouTube Monetization Analytics & Predictive Modeling Tool
    </h1>
    """,
    unsafe_allow_html=True
)
# --- Background color for main app only ---
main_bg_color = """
<style>
[data-testid="stAppViewContainer"] {
background-color: #F5F5F5; 
}
</style>
"""
# ✅ Apply CSS
st.markdown(main_bg_color, unsafe_allow_html=True)
# ---------------- Load Models ----------------
model_folder = r"D:\DS_Content_Monetization_Modeler\ML_Trained_Models"
ExtraTreesRegressor_model = joblib.load(f"{model_folder}/ExtraTreesRegressor.pkl")
KNeighborsRegressor_model = joblib.load(f"{model_folder}/KNeighborsRegressor_model.pkl")
DecisionTreeRegressor_model = joblib.load(f"{model_folder}/DecisionTreeRegressor_model.pkl")
RandomForestRegressor_model = joblib.load(f"{model_folder}/RandomForestRegressor_model.pkl")
GradientBoostingRegressor = joblib.load(f"{model_folder}/GradientBoostingRegressor.pkl")

# ---------------- Select Model ----------------
sb = st.selectbox(
    'ML Model',
    [
        "KNeighborsRegressor_model",
        "DecisionTreeRegressor_model",
        "RandomForestRegressor_model",
        "GradientBoostingRegressor_model",
        "ExtraTreesRegressor_model",
    ],
    index=0
)

model_dict = {
    "ExtraTreesRegressor_model": ExtraTreesRegressor_model,
    "KNeighborsRegressor_model": KNeighborsRegressor_model,
    "DecisionTreeRegressor_model": DecisionTreeRegressor_model,
    "RandomForestRegressor_model": RandomForestRegressor_model,
    "GradientBoostingRegressor_model": GradientBoostingRegressor
}

selected_model = model_dict[sb]

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("📌 Input Video Details")
views = st.sidebar.number_input("Views", min_value=0, step=1000, format="%d")
likes = st.sidebar.number_input("Likes", min_value=0, step=1000, format="%d")
comments = st.sidebar.number_input("Comments", min_value=0, step=100, format="%d")
watch_time_minutes = st.sidebar.number_input("Watch Time (minutes)", min_value=0.0, step=100.0, format="%.1f")
category = st.sidebar.selectbox("Category", ["Education", "Music", "Tech", "Entertainment", "Gaming", "Lifestyle"])
device = st.sidebar.selectbox("Device", ["TV", "Mobile", "Desktop", "Tablet"])

# Current date and time
now = datetime.now()
now = datetime.now()
engagement_rate = (likes + comments) / max(views, 1)  # Avoid division by zero
avg_watch_time_per_view = watch_time_minutes / max(views, 1)

# ---------------- Input DataFrame ----------------
input_data = pd.DataFrame({
    'views': [views],
    'likes': [likes],
    'comments': [comments],
    'watch_time_minutes': [watch_time_minutes],
    'video_length_minutes': [0],
    'subscribers': [0],
    'year': [now.year],
    'month': [now.month],
    'engagement_rate': [engagement_rate],
    'avg_watch_time_per_view': [avg_watch_time_per_view],
    'category_Entertainment': [1 if category=="Entertainment" else 0],
    'category_Gaming': [1 if category=="Gaming" else 0],
    'category_Lifestyle': [1 if category=="Lifestyle" else 0],
    'category_Music': [1 if category=="Music" else 0],
    'category_Tech': [1 if category=="Tech" else 0],
    'device_Mobile': [1 if device=="Mobile" else 0],
    'device_TV': [1 if device=="TV" else 0],
    'device_Tablet': [1 if device=="Tablet" else 0],
    'country_CA': [0],
    'country_DE': [0],
    'country_IN': [0],
    'country_UK': [0],
    'country_US': [0]
})

# ---------------- Display Inputs ----------------
st.write("### Video Details Entered")
st.write(f"**Views**: {views}")
st.write(f"**Likes**: {likes}")
st.write(f"**Comments**: {comments}")
st.write(f"**Watch Time**: {watch_time_minutes:.1f} min")
st.write(f"**Category**: {category}")
st.write(f"**Device**: {device}")

# ---------------- Prediction ----------------
if st.sidebar.button("🎯 Predict Revenue"):
    if views == 0:
        st.success("💰 Predicted Revenue: $0.00 USD")
    else:
        prediction = selected_model.predict(input_data)[0]
        st.success(f"💰 Predicted Revenue: ${prediction:.2f} USD")
# Make predictions for user input
predictions = []
for model_name, model in model_dict.items():
    pred = model.predict(input_data)[0]
    predictions.append({"Model": model_name, "Predicted Revenue (USD)": pred})

# Display in Streamlit
st.write("### Predicted Revenue by Model")
st.dataframe(predictions)