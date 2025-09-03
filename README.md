Diabetic Retinopathy Classification – NTI Project

📌 Overview
This project is part of the NTI training program.
Our main objective is to detect and classify diabetic retinopathy (DR) using retinal fundus images.
We applied one main computer vision task:

Classification to identify DR stage. Early detection of diabetic retinopathy can significantly improve patient outcomes, and this project demonstrates how AI can assist ophthalmologists in diagnosis.

📂 Dataset

Name: Diabetic Retinopathy Dataset

Images: Retinal fundus images

Classes: 5 DR stages – Healthy, Mild, Moderate, Severe, Proliferative DR

Sources: Kaggle / Public DR datasets

🛠 Workflow

0️⃣ Data Preprocessing

Input: RAW retinal fundus images

Steps: resizing, normalization, data augmentation (rotation, ZCA whitening, fill modes), train/val/test split

1️⃣ Classification (EfficientNetB3)

Task: Classify retinal images into 5 DR classes

Output: Predicted DR stage + probability

Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

2️⃣ Evaluation & Visualization

Measure: accuracy, precision, recall, F1-score

Confusion matrix and sample predictions visualized

Training/validation curves plotted for analysis

📊 Results

Task	Model	Metric	Score
Classification	EfficientNetB3	Accuracy	0.9927

*Replace with your actual test accuracy after training.

⚙️ Usage

Clone the repository.

Install dependencies:

pip install -r requirements.txt


Run the Streamlit app to test the model:

streamlit run app.py


Upload a retinal image to get DR classification results with probabilities.

💾 Model

The trained model is saved as: best-effB3-DR-model.h5.