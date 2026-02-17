
# 🧠 Simple ANN — Medical Insurance Cost Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![Type](https://img.shields.io/badge/Model-Regression-purple)

A simple Artificial Neural Network (ANN) built to predict **medical insurance charges** based on user health and demographic features.

This project demonstrates a complete machine-learning workflow — from data preprocessing to model training and evaluation.

---

## 📂 Repository Overview

```

📁 Simple_ANN_Project
│
├── 📓 Simple_ANN.ipynb    → ANN implementation notebook
├── 📄 insurance (1).csv    → Dataset
├── 📘 Mini Project_ Medical Insurance Cost Prediction with ANN.pdf
└── 📑 README.md

````

---

## 📌 Objective

Predict the **insurance cost** of individuals based on features such as:

- 👤 Age  
- 🚻 Gender  
- ⚖ BMI  
- 👶 Number of children  
- 🚬 Smoking status  
- 📍 Region  

---

## 🧪 Model Architecture (Overview)

- Input Layer  
- Hidden Dense Layers (ReLU)  
- Output Layer (Regression)

Built using **TensorFlow / Keras**

---

## ▶ Demo (Notebook Preview)

```python
model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, validation_split=0.2)
````

✔ Train — Evaluate — Predict — Visualize

(Full code inside the notebook)

---

## 📊 Results & Insights

* ANN successfully models the nonlinear relationship
* Smoking status & BMI highly impact insurance cost
* Neural networks perform well for regression tasks

Graphs include:

📉 Loss Curve
📈 Predicted vs Actual Charges

---

## 🧠 Learning Outcomes

✔ Data preprocessing
✔ Categorical encoding
✔ ANN design for regression
✔ Model performance evaluation

---

## 🚀 Future Scope

🔹 Hyperparameter tuning
🔹 Dropout / Regularization
🔹 Cross-validation
🔹 GUI / Web App deployment

---

## 🛠 Tech Stack

| Tool                 | Purpose        |
| -------------------- | -------------- |
| Python               | Programming    |
| Pandas / NumPy       | Data handling  |
| TensorFlow / Keras   | Neural Network |
| Matplotlib / Seaborn | Visualization  |
| Scikit-Learn         | ML utilities   |

---

## 📄 License

Open for learning & academic use.

---

## 👤 Author

**Parshva Mehta**

💻 Passionate about Data Science & AI

```

