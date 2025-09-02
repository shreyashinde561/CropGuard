# import streamlit as st
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # Load trained model
# model = load_model("crop_disease_model.h5")

# # Class labels (same as notebook)
# class_names = [
#     "Tomato___Bacterial_spot",
#     "Tomato___Early_blight",
#     "Tomato___Late_blight",
#     "Tomato___Leaf_Mold",
#     "Tomato___Septoria_leaf_spot",
#     "Tomato___Spider_mites",
#     "Tomato___Target_Spot",
#     "Tomato___YellowLeaf_Curl_Virus",
#     "Tomato___Mosaic_virus",
#     "Tomato___Healthy"
# ]

# st.title("🌱 Crop Disease Detection (Demo)")
# st.write("Upload a tomato leaf image and the model will predict the disease (demo using CIFAR-10).")

# # File uploader
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Load image
#     img = image.load_img(uploaded_file, target_size=(32, 32))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     # Prediction
#     predictions = model.predict(img_array)
#     class_index = np.argmax(predictions[0])
#     confidence = np.max(predictions[0])

#     # Display results
#     st.image(img, caption="Uploaded Image", use_container_width=True)
#     st.write(f"### 🔍 Predicted Disease: **{class_names[class_index]}**")
#     st.write(f"✅ Confidence: {confidence*100:.2f}%")
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from openai import OpenAI

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="🌱 Crop Disease Detection (Demo)",
    page_icon="🍅",
    layout="centered"
)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_my_model():
    return load_model("crop_disease_model.h5")

model = load_my_model()

# -------------------------
# Load OpenAI Client
# -------------------------
client = OpenAI(api_key=st.secrets["openai_api_key"])

# -------------------------
# Class labels (mapped from CIFAR-10)
# -------------------------
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___YellowLeaf_Curl_Virus",
    "Tomato___Mosaic_virus",
    "Tomato___Healthy"
]

# -------------------------
# App Title
# -------------------------
st.title("🌱 Crop Disease Detection (Demo)")
st.markdown("Upload a **tomato leaf image**. The AI will classify the disease and ChatGPT will suggest remedies.")

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Always resize to (32,32) because model was trained on CIFAR-10
    img = image.load_img(uploaded_file, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    disease_name = class_names[class_index]

    # Display Uploaded Image
    st.image(img, caption="🌿 Uploaded Leaf", use_container_width=True)

    # Prediction Result
    st.success(f"🔍 Predicted Disease: **{disease_name}**\n\n✅ Confidence: {confidence*100:.2f}%")

    # Probability Chart
    probs = predictions[0]
    df = pd.DataFrame({"Disease": class_names, "Probability": probs})
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
        x=alt.X("Probability", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Disease", sort='-x'),
        tooltip=["Disease", "Probability"]
    ).properties(height=400)
    st.markdown("### 📊 Prediction Confidence")
    st.altair_chart(chart, use_container_width=True)

    # -------------------------
    # Query ChatGPT for Remedies
    # -------------------------
    with st.spinner("💡 ChatGPT is generating remedies..."):
        prompt = f"The tomato leaf shows signs of: {disease_name}. Suggest causes, remedies, and prevention methods."
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        remedy_text = response.choices[0].message.content

    st.markdown("### 🧑‍🌾 Suggested Remedies")
    st.write(remedy_text)
