# 🩺 Diabetic Retinopathy Classification – NTI Project  

---

## 📌 Overview  
This project was developed as part of the **NTI training program**.  

The main objective is to **detect and classify diabetic retinopathy (DR)** from retinal fundus images using **deep learning**.  

Diabetic retinopathy is one of the leading causes of blindness. Early detection can significantly improve patient outcomes, and this project demonstrates how **AI-powered classification** can support ophthalmologists in diagnosis.  

---

## 📂 Dataset  
- **Name:** Diabetic Retinopathy Dataset  
- **Type:** Retinal fundus images  
- **Classes:**  
  - Healthy  
  - Mild  
  - Moderate  
  - Severe  
  - Proliferative DR  
- **Source:** [Kaggle](https://www.kaggle.com/datasets/sachinkumar413/diabetic-retinopathy-dataset/data)

---

## 🛠 Workflow  

### 0️⃣ Data Preprocessing  
- Input: Raw retinal fundus images  
- Steps:  
  - Resizing  
  - Normalization  
  - Data augmentation (rotation, ZCA whitening, fill modes)  
  - Train/validation/test split  

### 1️⃣ Classification (EfficientNetB3)  
- **Task:** Classify retinal images into 5 DR classes  
- **Output:** Predicted DR stage + probability  
- **Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix  

### 2️⃣ Evaluation & Visualization  
- Training/validation curves plotted  
- Confusion matrix generated  
- Sample predictions visualized  

---

## 📊 Results  
| Task          | Model           | Metric   | Score   |  
|---------------|----------------|----------|---------|  
| Classification | EfficientNetB3 | Accuracy | 0.9988 |  



---

## ⚙️ Usage  

### 🔹 Clone the repository  
```bash
git clone [https://github.com/your-username/diabetic-retinopathy-classification.git
cd diabetic-retinopathy-classification](https://github.com/Pant0x/diabetic-retinopathy-detection-NTI-project)


