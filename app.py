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

# Configure Streamlit page
st.set_page_config(page_title="Model Comparison Dashboard", page_icon="🔍", layout="wide")

def download_from_hf(url, filename):
    """Download model from Hugging Face if not already cached"""
    if not os.path.exists(filename):
        try:
            with st.spinner(f"Downloading {filename}..."):
                response = requests.get(url, timeout=300)  # 5 minute timeout
                response.raise_for_status()
                with open(filename, "wb") as f:
                    f.write(response.content)
                st.success(f"Successfully downloaded {filename}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download {filename}: {str(e)}")
            return False
    return True

@st.cache_resource
def load_model(path):
    """Load Keras model with error handling"""
    try:
        if not os.path.exists(path):
            st.error(f"Model file {path} not found")
            return None
            
        if path.endswith('.keras'):
            # Set memory growth to prevent GPU memory issues
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass  # GPU config might fail in some environments
                
            model = tf.keras.models.load_model(path, compile=False)
            return model
        else:
            st.warning(f"Cannot load {path} - only .keras files supported")
            return None
    except Exception as e:
        st.error(f"Error loading model from {path}: {str(e)}")
        return None

def predict_model(model, image, class_names):
    """Make prediction with performance metrics"""
    try:
        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Memory tracking (with fallback for systems without psutil)
        try:
            mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        except:
            mem_before = 0

        # Time the inference
        start_time = time.time()
        preds = model.predict(img_array, verbose=0)
        inference_time = time.time() - start_time

        # Memory after prediction
        try:
            mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            mem_used = max(0, mem_after - mem_before)  # Ensure non-negative
        except:
            mem_used = 0

        # Process predictions
        confidence = float(np.max(preds))
        pred_class = class_names[np.argmax(preds)]

        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'all_probabilities': preds[0].tolist(),
            'inference_time': inference_time,
            'memory_used': mem_used
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Model configuration
MODEL_INFOS = {
    "MobileNetV2": {
        "url": "https://huggingface.co/Bhavi23/MobilenetV2/resolve/main/multi_input_model_v1.keras",
        "filename": "mobilenetv2.keras",
        "description": "MobileNetV2 fine-tuned model for digit classification"
    }
}

CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

def create_performance_charts(results):
    """Create performance comparison charts"""
    if not results:
        return
        
    # Prepare data for plotting
    perf_df = pd.DataFrame([
        {
            "Model": k,
            "Confidence": v['confidence'],
            "Inference Time (s)": v['inference_time'],
            "Memory Used (MB)": v['memory_used']
        }
        for k, v in results.items()
    ])
    
    st.header("📈 Performance Summary")
    st.dataframe(perf_df, use_container_width=True)

    # Create charts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.barplot(data=perf_df, x='Model', y='Confidence', ax=ax1)
        ax1.set_title("Confidence by Model")
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.barplot(data=perf_df, x='Model', y='Inference Time (s)', ax=ax2)
        ax2.set_title("Inference Time by Model")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)
        plt.close(fig2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.barplot(data=perf_df, x='Model', y='Memory Used (MB)', ax=ax3)
        ax3.set_title("Memory Used by Model")
        ax3.tick_params(axis='x', rotation=45)
        st.pyplot(fig3)
        plt.close(fig3)

    return perf_df

def main():
    """Main application function"""
    st.title("🔍 Model Comparison Dashboard")
    st.markdown("Upload an image and compare predictions from multiple Keras models.")
    
    # Sidebar with model information
    with st.sidebar:
        st.header("📊 Models Info")
        for name, info in MODEL_INFOS.items():
            st.subheader(name)
            st.write(info['description'])
            st.write(f"Filename: {info['filename']}")
            st.divider()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image for classification...", 
        type=["png", "jpg", "jpeg"],
        help="Upload an image to classify using the available models"
    )

    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Download models if needed
            st.info("Checking and downloading models if needed...")
            models_ready = True
            for info in MODEL_INFOS.values():
                if not download_from_hf(info["url"], info["filename"]):
                    models_ready = False

            if not models_ready:
                st.error("Failed to download some models. Please refresh the page and try again.")
                return

            # Load models and make predictions
            results = {}
            progress_bar = st.progress(0)
            
            for i, (model_name, info) in enumerate(MODEL_INFOS.items()):
                st.info(f"Loading {model_name}...")
                model = load_model(info["filename"])
                
                if model is not None:
                    with st.spinner(f"Predicting with {model_name}..."):
                        res = predict_model(model, image, CLASS_NAMES)
                        if res is not None:
                            results[model_name] = res
                            st.success(f"✅ {model_name} prediction complete")
                        else:
                            st.error(f"❌ {model_name} prediction failed")
                else:
                    st.error(f"❌ Failed to load {model_name}")
                
                progress_bar.progress((i + 1) / len(MODEL_INFOS))

            # Display results if we have any
            if results:
                st.success("🎉 All predictions complete!")
                
                # Find best model
                best_model = max(results, key=lambda k: results[k]['confidence'])

                # Display individual model results
                st.header("🤖 Model Predictions")
                cols = st.columns(min(len(results), 3))
                
                for i, (model_name, res) in enumerate(results.items()):
                    with cols[i % len(cols)]:
                        is_best = model_name == best_model
                        
                        # Style the container
                        if is_best:
                            st.success(f"🏆 **{model_name}** (Best)")
                        else:
                            st.info(f"**{model_name}**")
                        
                        # Metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted Class", res['predicted_class'])
                            st.metric("Confidence", f"{res['confidence']:.4f}")
                        with col2:
                            st.metric("Inference Time", f"{res['inference_time']:.4f} s")
                            st.metric("Memory Used", f"{res['memory_used']:.2f} MB")
                        
                        # Confidence distribution
                        st.write("**Class Probabilities:**")
                        conf_df = pd.DataFrame({
                            'Class': CLASS_NAMES, 
                            'Probability': res['all_probabilities']
                        })
                        st.bar_chart(conf_df.set_index('Class'), height=200)

                # Performance comparison charts
                perf_df = create_performance_charts(results)

                # Recommendations
                if perf_df is not None and len(perf_df) > 1:
                    st.header("🎯 Model Recommendations")
                    
                    fastest = perf_df.loc[perf_df['Inference Time (s)'].idxmin()]
                    most_confident = perf_df.loc[perf_df['Confidence'].idxmax()]
                    most_efficient = perf_df.loc[perf_df['Memory Used (MB)'].idxmin()]

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⚡ Fastest Model", fastest['Model'], 
                                f"{fastest['Inference Time (s)']:.4f} s")
                    with col2:
                        st.metric("🎯 Most Confident", most_confident['Model'], 
                                f"{most_confident['Confidence']:.4f}")
                    with col3:
                        st.metric("💾 Most Memory Efficient", most_efficient['Model'], 
                                f"{most_efficient['Memory Used (MB)']:.2f} MB")

            else:
                st.error("No successful predictions were made. Please check the models and try again.")

        except Exception as e:
            st.error(f"An error occurred while processing the image: {str(e)}")
            st.error("Please try uploading a different image or refresh the page.")

    else:
        st.info("👆 Please upload an image to start the model comparison")
        
        # Show example of what the app does
        st.header("About this App")
        st.markdown("""
        This dashboard allows you to:
        - Upload an image for classification
        - Compare predictions from multiple trained models
        - View performance metrics (confidence, inference time, memory usage)
        - Get recommendations on which model to use for different scenarios
        
        **Supported image formats**: PNG, JPG, JPEG
        """)

if __name__ == "__main__":
    main()
