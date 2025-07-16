import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import psutil
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Model Comparison Dashboard", page_icon="🔍", layout="wide")

def download_from_hf(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename} ..."):
            response = requests.get(url)
            response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(response.content)

@st.cache_resource
def load_model(path):
    try:
        if path.endswith('.keras'):
            model = tf.keras.models.load_model(path)
            return model
        else:
            st.warning(f"Cannot load {path} - only .keras files supported")
            return None
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

def predict_model(model, image, class_names):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    start_time = time.time()

    preds = model.predict(img_array)

    inference_time = time.time() - start_time
    mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    mem_used = mem_after - mem_before

    confidence = np.max(preds)
    pred_class = class_names[np.argmax(preds)]

    return {
        'predicted_class': pred_class,
        'confidence': confidence,
        'all_probabilities': preds[0].tolist(),
        'inference_time': inference_time,
        'memory_used': mem_used
    }

MODEL_INFOS = {
    "MobileNetV2": {
        "url": "https://huggingface.co/Bhavi23/MobilenetV2/resolve/main/multi_input_model_v1.keras",
        "filename": "mobilenetv2.keras",
        "description": "MobileNetV2 fine-tuned model"
    }
}

CLASS_NAMES = ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9']

def main():
    st.title("🔍 Model Comparison Dashboard")
    st.markdown("Upload an image and compare predictions from multiple Keras models.")

    with st.sidebar:
        st.header("📊 Models Info")
        for name, info in MODEL_INFOS.items():
            st.subheader(name)
            st.write(info['description'])
            st.write(f"Filename: {info['filename']}")
            st.write("---")

    uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        for info in MODEL_INFOS.values():
            download_from_hf(info["url"], info["filename"])

        results = {}

        for model_name, info in MODEL_INFOS.items():
            model = load_model(info["filename"])
            if model is not None:
                with st.spinner(f"Predicting with {model_name}..."):
                    res = predict_model(model, image, CLASS_NAMES)
                    results[model_name] = res

        if results:
            best_model = max(results, key=lambda k: results[k]['confidence'])

            cols = st.columns(2)
            for i, (model_name, res) in enumerate(results.items()):
                with cols[i % 2]:
                    is_best = model_name == best_model
                    box_style = "background-color:#d4edda; border: 2px solid #28a745; border-radius: 10px; padding:10px;" if is_best else "border:1px solid #ddd; border-radius:10px; padding:10px;"
                    st.markdown(f"<div style='{box_style}'>", unsafe_allow_html=True)
                    st.markdown(f"### {model_name} {'🏆' if is_best else ''}")
                    st.metric("Predicted Class", res['predicted_class'])
                    st.metric("Confidence", f"{res['confidence']:.4f}")
                    st.metric("Inference Time", f"{res['inference_time']:.4f} s")
                    st.metric("Memory Used", f"{res['memory_used']:.2f} MB")
                    st.write("**Confidence Distribution:**")
                    conf_df = pd.DataFrame({'Class': CLASS_NAMES, 'Probability': res['all_probabilities']})
                    st.bar_chart(conf_df.set_index('Class'))
                    st.markdown("</div>", unsafe_allow_html=True)

            st.header("📈 Performance Summary")
            perf_df = pd.DataFrame([
                {
                    "Model": k,
                    "Confidence": v['confidence'],
                    "Inference Time (s)": v['inference_time'],
                    "Memory Used (MB)": v['memory_used']
                }
                for k,v in results.items()
            ])
            st.dataframe(perf_df)

            fig, ax = plt.subplots(figsize=(12,5))
            sns.barplot(data=perf_df, x='Model', y='Confidence', ax=ax)
            ax.set_title("Confidence by Model")
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(12,5))
            sns.barplot(data=perf_df, x='Model', y='Inference Time (s)', ax=ax2)
            ax2.set_title("Inference Time by Model")
            st.pyplot(fig2)

            fig3, ax3 = plt.subplots(figsize=(12,5))
            sns.barplot(data=perf_df, x='Model', y='Memory Used (MB)', ax=ax3)
            ax3.set_title("Memory Used by Model")
            st.pyplot(fig3)

            st.header("🎯 Recommendations")
            fastest = perf_df.loc[perf_df['Inference Time (s)'].idxmin()]
            most_confident = perf_df.loc[perf_df['Confidence'].idxmax()]
            most_efficient = perf_df.loc[perf_df['Memory Used (MB)'].idxmin()]

            cols = st.columns(3)
            cols[0].metric("⚡ Fastest Model", fastest['Model'], f"{fastest['Inference Time (s)']:.4f} s")
            cols[1].metric("🎯 Most Confident", most_confident['Model'], f"{most_confident['Confidence']:.4f}")
            cols[2].metric("💾 Most Memory Efficient", most_efficient['Model'], f"{most_efficient['Memory Used (MB)']:.2f} MB")

    else:
        st.info("👆 Please upload an image to start prediction")

if __name__ == "__main__":
    main()
