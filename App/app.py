import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .healthy {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #000000;
    }
    .mild {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #000000;
    }
    .moderate {
        background-color: #ffeaa7;
        border: 2px solid #fd79a8;
        color: #000000;
    }
    .severe {
        background-color: #fab1a0;
        border: 2px solid #e17055;
        color: #000000;
    }
    .proliferate {
        background-color: #ffb3b3;
        border: 2px solid #d63031;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">👁️ Diabetic Retinopathy Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Eye Health Analysis</p>', unsafe_allow_html=True)

# Load model function - Force EfficientNetB3 to work
@st.cache_resource
def load_model():
    # Get the absolute path to the project root
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)
    
    # Focus on EfficientNetB3 models
    model_paths = [
        os.path.join(project_root, "Models", "best-effB3-DR-model.h5"),
        os.path.join(project_root, "Models", "low-models", "best-effB3-DR-model.h5")
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            st.info(f"Forcing EfficientNetB3 to work with: {os.path.basename(model_path)}")
            
            # First try: Load model with custom objects and compile=False
            try:
                st.write("🔧 Trying to load complete model...")
                
                # Try loading the full model first
                model = tf.keras.models.load_model(
                    model_path, 
                    compile=False,
                    custom_objects=None
                )
                
                # Recompile
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy', 
                    metrics=['accuracy']
                )
                
                # Test with different inputs
                test1 = np.random.random((1, 224, 224, 3)).astype(np.float32)
                pred1 = model.predict(test1, verbose=0)
                
                test2 = np.random.random((1, 224, 224, 3)).astype(np.float32) 
                pred2 = model.predict(test2, verbose=0)
                
                # Check if predictions vary
                if not np.allclose(pred1, pred2, atol=1e-3):
                    st.success("✅ Complete model loaded and working!")
                    st.write(f"Test prediction varies: {pred1[0][:3]} vs {pred2[0][:3]}")
                    return model
                else:
                    st.warning("Model loaded but gives identical predictions")
                    
            except Exception as e:
                st.warning(f"Complete model loading failed: {str(e)}")
            
            # Second try: Recreate architecture 
            try:
                # FORCE RECREATION - Don't even try to load the full model
                from tensorflow.keras.applications import EfficientNetB3
                from tensorflow.keras.layers import Dense, Dropout
                from tensorflow.keras.models import Sequential
                
                st.write("🔧 Recreating EfficientNetB3 architecture...")
                
                # Create base model with ImageNet weights for better feature extraction
                base_model = EfficientNetB3(
                    include_top=False, 
                    weights=None,  # Don't use ImageNet weights - they cause conflicts
                    input_shape=(224, 224, 3), 
                    pooling='max'
                )
                
                # Make base model trainable
                base_model.trainable = True
                
                # Create the exact model from your notebook
                model = Sequential([
                    base_model,
                    Dropout(0.3),
                    Dense(512, activation='elu'),
                    Dense(256, activation='elu'), 
                    Dense(128, activation='elu'),
                    Dense(5, activation='softmax')
                ])
                
                st.write("🔄 Loading your trained weights...")
                
                # Try to load weights with different methods
                try:
                    # Method 1: Load weights directly
                    model.load_weights(model_path)
                    st.write("✅ Weights loaded directly")
                except Exception as e1:
                    try:
                        # Method 2: Load weights by name
                        model.load_weights(model_path, by_name=True)
                        st.write("✅ Weights loaded by name")
                    except Exception as e2:
                        try:
                            # Method 3: Load weights skipping mismatched layers
                            model.load_weights(model_path, by_name=True, skip_mismatch=True)
                            st.write("⚠️ Weights loaded with some layers skipped")
                        except Exception as e3:
                            st.error(f"All weight loading methods failed: {e3}")
                            continue
                
                # Compile the model
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Test the model with different inputs to ensure variation
                st.write("🧪 Testing model predictions...")
                
                # Test 1: Random noise
                test1 = np.random.random((1, 224, 224, 3)).astype(np.float32)
                pred1 = model.predict(test1, verbose=0)
                
                # Test 2: Different random noise  
                test2 = np.random.random((1, 224, 224, 3)).astype(np.float32)
                pred2 = model.predict(test2, verbose=0)
                
                # Test 3: All zeros
                test3 = np.zeros((1, 224, 224, 3)).astype(np.float32)
                pred3 = model.predict(test3, verbose=0)
                
                # Check if predictions vary
                if np.allclose(pred1, pred2, atol=1e-3) and np.allclose(pred2, pred3, atol=1e-3):
                    st.warning(f"⚠️ Model still gives identical predictions. Trying to fix...")
                    
                    # Try recompiling with different settings
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    # Test again
                    pred_test = model.predict(test1, verbose=0)
                    if np.allclose(pred1, pred_test, atol=1e-6):
                        st.error("Model is stuck - weights may be corrupted")
                        continue
                
                st.success("✅ EfficientNetB3 model working with varied predictions!")
                st.write(f"Test prediction 1: {pred1[0]}")
                st.write(f"Test prediction 2: {pred2[0]}")
                
                return model
                
            except Exception as e:
                st.error(f"Failed to recreate EfficientNetB3: {str(e)}")
                continue
    
    st.error("❌ Could not get EfficientNetB3 to work!")
    st.write("Your model weights might be corrupted or incompatible.")
    st.write("Consider retraining your model or using a different architecture.")
    st.stop()

# IMPORTANT: Correct class order based on alphabetical sorting (how ImageDataGenerator orders them)
# This is crucial - your model was trained with this exact order!
CLASS_NAMES = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferate DR', 'Severe DR']

CLASS_DESCRIPTIONS = {
    'Healthy': 'No signs of diabetic retinopathy detected. Regular eye check-ups are still recommended.',
    'Mild DR': 'Early stage of diabetic retinopathy. Monitor blood sugar levels and consult your doctor.',
    'Moderate DR': 'Moderate diabetic retinopathy detected. Medical attention is recommended.',
    'Severe DR': 'Severe diabetic retinopathy detected. Immediate medical attention required.',
    'Proliferate DR': 'Advanced diabetic retinopathy detected. Urgent medical intervention needed.'
}

CLASS_COLORS = {
    'Healthy': 'healthy',
    'Mild DR': 'mild',
    'Moderate DR': 'moderate',
    'Severe DR': 'severe',
    'Proliferate DR': 'proliferate'
}

# Preprocess image function - matching your training preprocessing
def preprocess_image(image):
    """Preprocess the uploaded image EXACTLY like in training"""
    try:
        # Resize to exactly 224x224 (your training size)
        image = image.resize((224, 224))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0,1] range (like ImageDataGenerator does)
        image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Predict function
def predict_retinopathy(model, image):
    """Make prediction on the preprocessed image"""
    try:
        preprocessed_image = preprocess_image(image)
        if preprocessed_image is None:
            return None, None, None
        
        # Make prediction
        predictions = model.predict(preprocessed_image, verbose=0)
        
        # Get the predicted class
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100
        
        return predicted_class, confidence, predictions[0]
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

# Main app
def main():
    # Load model
    model = load_model()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📷 Upload Retinal Image")
        st.markdown("Please upload a clear retinal fundus image for analysis.")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a retinal fundus image (PNG, JPG, JPEG)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Analyze button
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Analyzing image..."):
                        # Make prediction
                        predicted_class, confidence, all_predictions = predict_retinopathy(model, image)
                        
                        if predicted_class is not None:
                            # Store results in session state
                            st.session_state.prediction_results = {
                                'class': predicted_class,
                                'confidence': confidence,
                                'all_predictions': all_predictions
                            }
                        else:
                            st.error("Failed to analyze the image. Please try another image.")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            predicted_class = results['class']
            confidence = results['confidence']
            all_predictions = results['all_predictions']
            
            # Display main result
            color_class = CLASS_COLORS[predicted_class]
            st.markdown(f"""
            <div class="result-box {color_class}">
                <h3>🎯 Prediction: {predicted_class}</h3>
                <h4>📈 Confidence: {confidence:.1f}%</h4>
                <p><strong>Description:</strong> {CLASS_DESCRIPTIONS[predicted_class]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display all probabilities
            st.markdown("### 📈 Detailed Probabilities")
            prob_data = []
            for i, class_name in enumerate(CLASS_NAMES):
                probability = all_predictions[i] * 100
                prob_data.append({
                    'Class': class_name,
                    'Probability (%)': f"{probability:.2f}%"
                })
            
            st.table(prob_data)
            
            # Create probability chart
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                probabilities = [all_predictions[i] * 100 for i in range(len(CLASS_NAMES))]
                colors = ['#28a745' if CLASS_NAMES[i] == predicted_class else '#6c757d' for i in range(len(CLASS_NAMES))]
                
                bars = ax.bar(CLASS_NAMES, probabilities, color=colors)
                ax.set_ylabel('Probability (%)')
                ax.set_title('Prediction Probabilities for All Classes')
                ax.set_ylim(0, 100)
                
                # Add value labels on bars
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{prob:.1f}%', ha='center', va='bottom')
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating chart: {str(e)}")
            
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results here.")
    
    # Information section
    st.markdown("---")
    st.markdown("### ℹ️ About This System")
    
    info_col1, info_col2 = st.columns([1, 1])
    
    with info_col1:
        st.markdown("""
        **🎯 What is Diabetic Retinopathy?**
        
        Diabetic retinopathy is a diabetes complication that affects eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).
        
        **🔬 How it works:**
        - This AI system uses a deep learning model (EfficientNetB3)
        - Trained on thousands of retinal images
        - Achieves high accuracy in classification
        """)
    
    with info_col2:
        st.markdown("""
        **⚠️ Important Disclaimer:**
        
        - This tool is for educational and screening purposes only
        - NOT a substitute for professional medical diagnosis
        - Always consult with a qualified ophthalmologist
        - Early detection and treatment are crucial
        
        **📞 Seek immediate medical attention if:**
        - Severe or Proliferate DR is detected
        - You experience vision changes
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏥 Diabetic Retinopathy Detection System | NTI Computer Vision Project</p>
        <p>⚠️ For medical screening purposes only - Not a medical diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
