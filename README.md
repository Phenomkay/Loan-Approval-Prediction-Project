### Loan Approval Prediction Web Application

---

## 📌 Project Title: **Loan Approval Prediction**

---

## 📍 Project Objective:
The objective of this project is to build a machine learning model that predicts whether a loan application will be approved or rejected based on applicant information. The model is trained using a Random Forest Classifier and deployed as a web application using **Streamlit** to provide a user-friendly interface for predicting loan approval.

---

## 📄 Project Overview:
Loan approval decisions are critical for financial institutions. The goal of this project is to automate this process by leveraging machine learning techniques. The web app allows users to input key features such as income, loan amount, assets, and more, to predict whether a loan will be approved or rejected.

The project follows the following stages:
1. **Exploratory Data Analysis**
2. **Data Preprocessing**
3. **Model Training using Random Forest**
4. **Evaluation of Model Performance**
5. **Deployment using Streamlit Web App**

---

## 🔧 Technologies Used:
- **Programming Language**: Python
- **Libraries**:
  - `pandas`, `numpy`: Data manipulation and analysis.
  - `scikit-learn`: Machine learning algorithms (Random Forest Classifier).
  - `pickle`: Saving the trained model, scaler, and encoder for future use.
  - `streamlit`: Web app deployment.
  - `matplotlib`, `seaborn`: Data visualization.
  - `base64`: Image encoding for background setting.
  
---

## 📂 Dataset:
The dataset contains various features related to loan applications. Key columns include:
- `loan_id`: Loan identification number.
- `no_of_dependents`: Number of dependents the applicant has.
- `education`: Applicant’s education level (Graduate/Not Graduate).
- `self_employed`: Whether the applicant is self-employed or not.
- `income_annum`: The applicant’s annual income.
- `loan_amount`: The loan amount requested by the applicant.
- `loan_term`: The duration of the loan in months.
- `cibil_score`: The applicant’s CIBIL credit score (ranging from 300 to 900).
- `residential_assets_value`: Value of the applicant's residential assets.
- `commercial_assets_value`: Value of the applicant's commercial assets.
- `luxury_assets_value`: Value of the applicant's luxury assets.
- `bank_asset_value`: Value of the applicant's bank assets.
- `loan_status`: The target variable, indicating whether the loan was approved or rejected.

---

The project follows the following stages:

##  Exploratory Data Analysis:  
   During EDA, it was discovered that the dataset contains unnecessary space characters in both column names and string data. To handle this:
   - The column names were stripped of any leading or trailing spaces using the `str.strip()` method.
   - Similarly, string columns were processed using `applymap(lambda x: x.strip())` to remove extra spaces within the data entries.
   This ensured cleaner data, preventing potential issues during model training.

---

## 🔍 Data Preprocessing:

### 1. **Removing Unnecessary Columns:**
   The `loan_id` column was dropped as it does not provide useful information for prediction.

### 2. **Handling Categorical Data:**
   - The `education` and `self_employed` columns were label encoded using `LabelEncoder`.
   - The target variable `loan_status` was mapped to binary values (`Approved` = 1, `Rejected` = 0).

### 3. **Feature Scaling:**
   - Feature scaling was performed using `StandardScaler` to normalize the feature values and improve the performance of the machine learning model.

---

## ⚙️ Model Training:

A **Random Forest Classifier** was chosen due to its robustness and ability to handle a mix of numerical and categorical data. The following steps were performed:

### 1. **Train-Test Split**:
   The dataset was split into 80% training and 20% testing using `train_test_split` from scikit-learn.

### 2. **Training the Random Forest Model**:
   - A Random Forest Classifier was instantiated with 20 trees (`n_estimators=20`) and entropy as the splitting criterion (`criterion='entropy'`).
   - The model was trained on the training dataset using `rf_model.fit(x_train, y_train)`.

### 3. **Evaluation**:
   The model was evaluated using the test set and achieved an accuracy score of **97%**.

---

## 💾 Saving the Model:

To deploy the model, the trained classifier, encoder, and scaler were saved using the `pickle` module.

```python
pickle.dump(rf_model, open('model.pkl', 'wb'))
pickle.dump(le, open('encoder.pkl', 'wb'))
pickle.dump(sc, open('scaler.pkl', 'wb'))
```

---

## 🌐 Web App Deployment:

The web application was deployed using **Streamlit**, a fast way to build and share data science applications. The app provides a user-friendly interface for predicting whether a loan will be approved based on the following features:

- Number of Dependents
- Education Level
- Self Employment Status
- Applicant's Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Value of Residential Assets
- Value of Commercial Assets
- Value of Luxury Assets
- Value of Bank Assets

### Key Features of the Web App:
- **Background Image**: A background image related to loans was set using `base64` encoding for enhanced user experience.
- **User Inputs**: The user can input various loan details, and upon pressing the "Predict Loan Status" button, the model will predict whether the loan will be approved or rejected.
- **Prediction Output**: The prediction result is displayed as a **success** message for approved loans or an **error** message for rejected loans.

### Code to Start the Web App:

```bash
streamlit run app.py
```

---

## 🤝 Contributions:

Feel free to open a pull request or raise an issue if you want to contribute to the project. All contributions are welcome!

---

## 🙌 Acknowledgments:

- **Scikit-learn**: For providing a simple and efficient tool for data mining and data analysis.
- **Streamlit**: For simplifying the deployment of machine learning models with minimal effort.
- **Pandas & Numpy**: For data manipulation and analysis tools.

---

## 🧑‍💻 Author:

- **[Caleb Osagie]**  
  Connect with me on [LinkedIn]([https://linkedin.com/in/yourprofile](https://www.linkedin.com/in/caleb-osagie-37a793123/)).
