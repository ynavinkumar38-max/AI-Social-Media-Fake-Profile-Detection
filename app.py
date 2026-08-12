import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from PIL import Image
from datetime import datetime

from image_classifier import classify_image

st.set_page_config(
    page_title="AI Fake Profile Detection",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_models():
    instagram_model = joblib.load("model/instagram_model.pkl")
    facebook_model = joblib.load("model/facebook_model.pkl")

    return instagram_model, facebook_model


instagram_model, facebook_model = load_models()

if not os.path.exists("history"):
    os.makedirs("history")

history_file = "history/history.json"

if not os.path.exists(history_file):
    with open(history_file, "w") as f:
        json.dump([], f)

def save_history(data):

    with open(history_file, "r") as f:
        history = json.load(f)

    history.append(data)

    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

def load_history():

    with open(history_file, "r") as f:
        return json.load(f)

st.markdown("""
<style>

.main-title{
font-size:42px;
font-weight:bold;
color:#2E86C1;
text-align:center;
}

.sub-title{
font-size:18px;
text-align:center;
color:gray;
margin-bottom:20px;
}

.result-box{

padding:20px;
border-radius:15px;
background:#f4f6f7;

}

.metric-card{

padding:15px;
border-radius:10px;
background:white;
box-shadow:0px 0px 10px rgba(0,0,0,0.1);

}

</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='main-title'>🤖 AI Social Media Fake Profile Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Detect Fake, Suspicious and Real Social Media Profiles using AI</div>",
    unsafe_allow_html=True
)

st.markdown("---")

st.sidebar.title("⚙ Settings")

platform = st.sidebar.selectbox(

    "Select Platform",

    [

        "Instagram",

        "Facebook"

    ]

)

show_history = st.sidebar.checkbox(

    "Show History"

)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Developed using

✅ Streamlit

✅ TensorFlow

✅ Scikit-Learn

✅ MobileNetV2

"""
)

st.header("📷 Upload Profile Picture")

uploaded_file = st.file_uploader(
    "Upload Profile Image",
    type=["jpg", "jpeg", "png"]
)

image = None
image_type = "Unknown"

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:

        st.image(
            image,
            caption="Uploaded Profile",
            use_container_width=True
        )

    with col2:

        st.subheader("🤖 AI Image Analysis")

        with st.spinner("Analyzing Image..."):

            image_type = classify_image(image)

        if image_type == "Human":
            st.success("✅ Human Face Detected")

        elif image_type == "Animal":
            st.warning("🐶 Animal Image")

        elif image_type == "Vehicle":
            st.warning("🚗 Vehicle Image")

        elif image_type == "Nature":
            st.info("🌳 Nature Image")

        elif image_type == "Cartoon":
            st.warning("🎭 Cartoon / Anime Image")

        elif image_type == "Logo":
            st.info("🖼 Logo Image")

        else:
            st.error("❓ Unknown Image")

        st.metric(
            "Detected Image Type",
            image_type
        )

st.markdown("---")

st.header("👤 Profile Information")

left, right = st.columns(2)

with left:

    username = st.text_input(
        "Username"
    )

    followers = st.number_input(
        "Followers",
        min_value=0,
        value=100
    )

    following = st.number_input(
        "Following",
        min_value=0,
        value=50
    )

    posts = st.number_input(
        "Posts",
        min_value=0,
        value=10
    )

with right:

    verified = st.selectbox(
        "Verified Account",
        ["No", "Yes"]
    )

    bio = st.selectbox(
        "Bio Available",
        ["No", "Yes"]
    )

    profile_pic = st.selectbox(
        "Profile Picture",
        ["Yes", "No"]
    )

st.markdown("---")

verified_value = 1 if verified == "Yes" else 0
bio_value = 1 if bio == "Yes" else 0
profile_pic_value = 1 if profile_pic == "Yes" else 0

username_length = len(username)

username_digits = sum(
    c.isdigit()
    for c in username
)

if following == 0:
    ratio = followers
else:
    ratio = followers / following

features = pd.DataFrame([{

    "followers": followers,

    "following": following,

    "posts": posts,

    "verified": verified_value,

    "bio": bio_value,

    "profile_pic": profile_pic_value,

    "username_length": username_length,

    "username_digits": username_digits,

    "ratio": ratio

}])

st.header("🚀 Start Analysis")

analyze = st.button(
    "🔍 Analyze Profile",
    use_container_width=True
)

st.markdown("---")

if analyze:

    if uploaded_file is None:
        st.error("Please upload a profile image.")
        st.stop()

    if username.strip() == "":
        st.error("Please enter username.")
        st.stop()

    if platform == "Instagram":
        model = instagram_model
    else:
        model = facebook_model

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    confidence = round(max(probability) * 100, 2)

    trust_score = 50
    reasons = []

    if image_type == "Human":
        trust_score += 20
        reasons.append("✅ Human profile image detected")

    elif image_type == "Cartoon":
        trust_score -= 10
        reasons.append("⚠ Cartoon profile image")

    elif image_type == "Logo":
        trust_score -= 5
        reasons.append("ℹ Logo profile image")

    elif image_type == "Animal":
        trust_score -= 20
        reasons.append("⚠ Animal profile image")

    elif image_type == "Vehicle":
        trust_score -= 15
        reasons.append("⚠ Vehicle profile image")

    elif image_type == "Nature":
        trust_score -= 10
        reasons.append("⚠ Nature image used as profile")

    else:
        trust_score -= 15
        reasons.append("⚠ Unknown image")

    # Metadata
    if verified_value:
        trust_score += 20
        reasons.append("✅ Verified account")

    if bio_value:
        trust_score += 10
        reasons.append("✅ Bio available")
    else:
        trust_score -= 10
        reasons.append("⚠ No bio")

    if posts >= 20:
        trust_score += 10
        reasons.append("✅ Active profile")
    else:
        trust_score -= 5
        reasons.append("⚠ Very few posts")

    if ratio >= 1:
        trust_score += 10
        reasons.append("✅ Healthy follower ratio")
    else:
        trust_score -= 10
        reasons.append("⚠ Poor follower ratio")

    if username_digits >= 5:
        trust_score -= 10
        reasons.append("⚠ Username contains many digits")

    if prediction == 1:
        trust_score += 10
        reasons.append("✅ Machine Learning predicts Real")
    else:
        trust_score -= 20
        reasons.append("⚠ Machine Learning predicts Fake")

    trust_score = max(0, min(100, trust_score))
    risk_score = 100 - trust_score

    if trust_score >= 75:
        final_result = "🟢 REAL PROFILE"

    elif trust_score >= 45:
        final_result = "🟡 SUSPICIOUS PROFILE"

    else:
        final_result = "🔴 FAKE PROFILE"

    st.header("📊 Analysis Result")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Trust Score", f"{trust_score}%")

    with c2:
        st.metric("Risk Score", f"{risk_score}%")

    with c3:
        st.metric("ML Confidence", f"{confidence}%")

    st.success(final_result)

    st.subheader("Reasons")

    for item in reasons:
        st.write(item)

    save_history({

        "date": datetime.now().strftime("%d-%m-%Y %H:%M"),

        "platform": platform,

        "username": username,

        "image_type": image_type,

        "trust_score": trust_score,

        "risk_score": risk_score,

        "prediction": final_result

    })
  
if show_history:

    st.header("📜 Analysis History")

    history = load_history()

    if len(history) == 0:

        st.info("No history available.")

    else:

        df = pd.DataFrame(history)

        st.dataframe(
            df,
            use_container_width=True
        )

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download History (CSV)",
            data=csv,
            file_name="analysis_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("🗑 Clear History", use_container_width=True):

            with open(history_file, "w") as f:
                json.dump([], f)

            st.success("History cleared successfully.")
            st.rerun()

st.markdown("---")

history = load_history()

if len(history) > 0:

    st.header("📊 Analytics Dashboard")

    df = pd.DataFrame(history)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Total Profiles",
            len(df)
        )

    with c2:
        st.metric(
            "Average Trust Score",
            f"{round(df['trust_score'].mean(),1)}%"
        )

    with c3:
        fake_count = (
            df["prediction"]
            .str.contains("FAKE")
            .sum()
        )

        st.metric(
            "Fake Profiles",
            fake_count
        )

st.markdown("---")

if analyze:

    report = f"""
AI SOCIAL MEDIA PROFILE REPORT

Date : {datetime.now().strftime("%d-%m-%Y %H:%M")}

Platform : {platform}

Username : {username}

Image Type : {image_type}

Followers : {followers}

Following : {following}

Posts : {posts}

Verified : {verified}

Bio : {bio}

Trust Score : {trust_score}%

Risk Score : {risk_score}%

Prediction :

{final_result}

Reasons

"""

    for reason in reasons:
        report += f"\n{reason}"

    st.download_button(

        "📄 Download Report",

        report,

        file_name=f"{username}_report.txt",

        mime="text/plain",

        use_container_width=True

    )

st.markdown("---")

st.markdown(
"""
<center>

### 🤖 AI Social Media Fake Profile Detection System

Developed using

**Streamlit | TensorFlow | MobileNetV2 | Scikit-Learn**

Final Year Project

</center>
""",
unsafe_allow_html=True
)