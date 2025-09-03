# Diabetic Retinopathy Detection App

A user-friendly web application for detecting diabetic retinopathy using AI.

## Features

- 🔍 **Easy Image Upload**: Simply drag and drop or browse for retinal images
- 🎯 **AI Prediction**: Uses your trained EfficientNetB3 model for accurate classification
- 📊 **Detailed Results**: Shows confidence scores and probabilities for all classes
- 🎨 **User-Friendly Interface**: Clean and intuitive design
- 📈 **Visualization**: Interactive charts showing prediction probabilities

## Classes Detected

1. **Healthy** - No signs of diabetic retinopathy
2. **Mild DR** - Early stage diabetic retinopathy
3. **Moderate DR** - Moderate diabetic retinopathy
4. **Severe DR** - Severe diabetic retinopathy
5. **Proliferate DR** - Advanced diabetic retinopathy

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run app.py
   ```

3. **Open in Browser**: The app will automatically open in your default browser at `http://localhost:8501`

## Usage

1. Upload a retinal fundus image (PNG, JPG, JPEG)
2. Click "Analyze Image" 
3. View the prediction results and confidence scores
4. See detailed probabilities for all classes

## Important Notes

⚠️ **Medical Disclaimer**: This application is for educational and screening purposes only. It is NOT a substitute for professional medical diagnosis. Always consult with a qualified ophthalmologist for proper medical evaluation.

## Model Information

- **Architecture**: EfficientNetB3 with transfer learning
- **Input Size**: 224x224 pixels
- **Classes**: 5 categories of diabetic retinopathy
- **Model File**: Uses `best-effB3-DR-model.h5` from the Models directory
