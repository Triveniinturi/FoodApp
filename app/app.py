import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Calorie Data ----------
calories_per_100g = {
    "AlooMasala": 100, "Bhatura": 355, "BhindiMasala": 107, "Biryani": 170,
    "Chai": 50, "Chole": 148, "CoconutChutney": 402, "Dal": 116,
    "Dosa": 150, "DumAloo": 170, "FishCurry": 124, "Ghevar": 700,
    "GreenChutney": 175, "GulabJamun": 370, "Idli": 90, "Jalebi": 356,
    "Kebab": 250, "Kheer": 120, "Kulfi": 224, "Lassi": 80,
    "MuttonCurry": 120, "OnionPakoda": 464, "PalakPaneer": 169,
    "Poha": 130, "RajmaCurry": 130, "RasMalai": 120, "Samosa": 362,
    "ShahiPaneer": 175, "WhiteRice": 111
}

# default serving grams for each dish
default_serving_grams = {k: 150 for k in calories_per_100g.keys()}
default_serving_grams.update({
    "Bhatura": 100, "Chai": 200, "Idli": 50, "Lassi": 200, "CoconutChutney": 30
})

# healthier replacements
healthy_replacements = {
    "AlooMasala": ["Mixed Vegetable Curry", "Steamed Potato with Herbs"],
    "Bhatura": ["Whole-Wheat Chapati", "Baked Bhatura"],
    "Biryani": ["Vegetable Pulao with Brown Rice", "Quinoa Biryani"],
    "Chole": ["Boiled Chickpea Salad", "Lite Chole (low oil)"],
    "Samosa": ["Baked Samosa", "Veggie Spring Roll (air-fried)"],
    "ShahiPaneer": ["Paneer Tikka", "Palak Paneer (low-fat)"],
    "GulabJamun": ["Dry Fruit Ladoo", "Yogurt + Honey Dessert"],
    "MuttonCurry": ["Grilled Chicken Tikka", "Fish Curry (tomato base)"],
    "WhiteRice": ["Brown Rice", "Quinoa"],
    "Dal": ["Mixed Dal + Vegetables", "Sprouted Moong Dal"],
}

# ---------- Helper Functions ----------
def estimate_calories(label, grams=100):
    kcal_100 = calories_per_100g.get(label, 150)
    return kcal_100 * grams / 100

def compute_health_score(label):
    kcal_100 = calories_per_100g.get(label, 150)
    score = 50
    if kcal_100 >= 400: score -= 25
    elif kcal_100 >= 300: score -= 15
    elif kcal_100 >= 200: score -= 5
    if label in ["Dal", "Chole", "Poha", "Idli", "Dosa", "RajmaCurry"]: score += 15
    if label in ["Samosa", "Bhatura", "GulabJamun", "OnionPakoda"]: score -= 20
    return max(0, min(100, score))

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="Food Calorie Recommender")
st.title("🍱 Indian Food Calorie Estimator & Healthy Recommender")

uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
weights_path = "/content/drive/MyDrive/Colab Notebooks/FoodDetection/Food-Calorie-Recommender-app/yolov8/best.pt"

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model = YOLO(weights_path)
    with st.spinner("Detecting food items..."):
        results = model.predict(source=np.array(image), imgsz=640, conf=0.1, verbose=False)
    res = results[0]
    boxes = res.boxes
    names = model.names
    detections = []

    if len(boxes) == 0:
        st.warning("No food items detected.")
    else:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = names.get(cls, str(cls))
            kcal = estimate_calories(label, default_serving_grams.get(label, 100))
            detections.append({
                "Label": label, "Confidence": round(conf,2),
                "Calories": round(kcal,1),
                "Health Score": compute_health_score(label)
            })

        df = pd.DataFrame(detections)
        total = df["Calories"].sum()
        st.subheader("🍽️ Detected Foods")
        st.dataframe(df)
        st.metric("Estimated Total Calories", f"{total:.0f} kcal")

        # Display bounding boxes
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(image)
        for d, box in zip(detections, boxes):
            coords = box.xyxy[0]
            if coords.is_cuda:
              coords = coords.cpu()
            x1, y1, x2, y2 = coords.numpy()
            rect = plt.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"{d['Label']} ({d['Calories']:.0f} kcal)", color='yellow', fontsize=9, weight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))
        ax.axis("off")
        st.pyplot(fig)

        # Health recommendations
        st.header("💡 Healthier Alternatives")
        for d in detections:
            label = d["Label"]
            recs = healthy_replacements.get(label, [])
            st.subheader(f"{label} → {', '.join(recs) if recs else 'No alternatives listed'}")
            st.write(f"Health Score: {d['Health Score']}/100")
