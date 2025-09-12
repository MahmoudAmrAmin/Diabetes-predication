# 🩺 Diabetes Prediction using Machine Learning  

## 📌 Project Overview  
This project demonstrates how **machine learning models** can be applied to predict diabetes using the Pima Indians Diabetes Dataset.  
The notebook walks through the **end-to-end ML pipeline**, from exploratory data analysis (EDA) to preprocessing, model training, and evaluation.  

It compares the performance of multiple classifiers, addresses data imbalance with **SMOTE**, and evaluates results using robust metrics.  

---

## ⚙️ Installation  

1. **Clone this repository**  
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

2. **Create and activate a virtual environment**  
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install dependencies**  
```bash
pip install -r requirements.txt
```

4. **Run Jupyter Notebook**  
```bash
jupyter notebook
```

---

## 🚀 Usage  

1. Launch Jupyter Notebook and open:  
   ```Diabetesclassification.ipynb```  

2. Run the notebook step by step:  
   - 📊 **Exploratory Data Analysis (EDA)** – visualize distributions, correlations, and missing values  
   - ⚙️ **Preprocessing** – handle missing values, scale data, balance classes using **SMOTE**  
   - 🤖 **Model Training** – fit multiple classifiers:  
     - Logistic Regression  
     - Perceptron  
     - Decision Tree  
     - Random Forest  
     - AdaBoost  
     - Bagging  
     - Gradient Boosting  
     - Support Vector Machine (SVC)  
     - K-Nearest Neighbors (KNN)  
     - XGBoost  
     - CatBoost  
   - 📈 **Evaluation** – compare models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC  

3. **Example: Predict diabetes for a new patient**  
```python
sample_data = [[148, 72, 35, 0, 33.6, 0.627, 50]]  # Example patient data
prediction = model.predict(sample_data)
print("Diabetes Risk:", "Yes" if prediction[0] == 1 else "No")
```

---

## 📊 Dataset  

The dataset used is the **Pima Indians Diabetes Dataset**.  

**Features:**  
- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  

**Target:**  
- Outcome → `1` = Positive for diabetes, `0` = Negative  

---

## 📦 Dependencies  

- Python ≥ 3.8  
- pandas, numpy  
- seaborn, matplotlib  
- scikit-learn  
- imbalanced-learn (SMOTE)  
- xgboost  
- catboost  

Install them via:  
```bash
pip install -r requirements.txt
```

---

## 📈 Results  

The notebook compares all models and highlights the best-performing one based on **ROC-AUC** and other metrics.  
Visualizations (confusion matrices, ROC curves, bar plots) make results easy to interpret.  

---

## 🤝 Contribution  

Contributions are welcome!  

1. Fork this repo  
2. Create a new branch (`feature/your-feature`)  
3. Commit your changes  
4. Push and open a Pull Request  

---

## 📜 License  
This project is licensed under the MIT License.  

