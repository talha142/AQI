# Import necessary libraries
import streamlit as st  # For creating the web app interface
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import seaborn as sns  # For statistical data visualization
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # For model evaluation
from sklearn.linear_model import LogisticRegression  # Logistic Regression classifier
from sklearn.tree import DecisionTreeClassifier  # Decision Tree classifier
from sklearn.naive_bayes import GaussianNB  # Naive Bayes classifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier
from xgboost import XGBClassifier  # XGBoost classifier
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables

# Load Dataset
@st.cache_data  # Cache the data to improve performance (Streamlit won't reload data unnecessarily)
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("city_day.csv")
    return data

# Data Cleaning
def clean_data(data):
    numerical_columns = data.select_dtypes(include=['number']).columns
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())
    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
    data.drop_duplicates(inplace=True)
    Q1 = data[numerical_columns].quantile(0.25)
    Q3 = data[numerical_columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    for column in numerical_columns:
        data = data[(data[column] >= lower_bound[column]) & (data[column] <= upper_bound[column])]
    return data

# Encode Target Variable
def encode_target(data):
    label_encoder = LabelEncoder()
    data['AQI_Bucket_Encoded'] = label_encoder.fit_transform(data['AQI_Bucket'])
    return data, label_encoder

# Home Page
def home_page():
    st.title("🌍 Air Quality Index (AQI) Prediction and Analysis")
    st.write("Welcome to the AQI Project! This application helps you understand, analyze, and predict air quality using machine learning models.")
    st.header("❓ What is AQI?")
    st.write("""
    The **Air Quality Index (AQI)** is a measure used to communicate how polluted the air currently is or how polluted it is forecast to become. 
    It is calculated based on the concentrations of major air pollutants, such as:
    - **PM2.5** (Particulate Matter 2.5 micrometers or smaller)
    - **PM10** (Particulate Matter 10 micrometers or smaller)
    - **NO2** (Nitrogen Dioxide)
    - **SO2** (Sulfur Dioxide)
    - **CO** (Carbon Monoxide)
    - **O3** (Ozone)
    """)

# EDA Page
def eda_page(data):
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.write("### Dataset Overview")
    st.write(data.head(10))

    st.write("### Basic Statistics")
    st.write(data.describe())

    st.write("### Correlation Heatmap")
    corr_matrix = data[['PM2.5', 'PM10', 'NO', 'SO2', 'O3', 'CO', 'AQI']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    st.write("### Average AQI per AQI Category")
    group_by = data.groupby("AQI_Bucket")["AQI"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=group_by.index, y=group_by.values, palette="coolwarm", ax=ax)
    plt.xlabel("AQI Category")
    plt.ylabel("Average AQI")
    plt.title("Average AQI per AQI Category")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Prediction Page
def prediction_page(data, label_encoder):
    st.title("🔮 AQI Prediction")
    st.write("### Select Features for Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("#### Pollutant Levels")
        PM2_5 = st.number_input("PM2.5", value=data["PM2.5"].mean(), step=0.1)
        PM10 = st.number_input("PM10", value=data["PM10"].mean(), step=0.1)

    with col2:
        st.write("#### Pollutant Levels")
        NO = st.number_input("NO", value=data["NO"].mean(), step=0.1)
        SO2 = st.number_input("SO2", value=data["SO2"].mean(), step=0.1)

    with col3:
        st.write("#### Pollutant Levels")
        O3 = st.number_input("O3", value=data["O3"].mean(), step=0.1)
        CO = st.number_input("CO", value=data["CO"].mean(), step=0.1)

    st.write("### Select Model")
    model_option = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Naive Bayes", "Random Forest", "XGBoost"])

    if st.button("Predict"):
        X = data[["PM2.5", "PM10", "NO", "SO2", "O3", "CO"]]
        y = data["AQI_Bucket_Encoded"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_option == "Naive Bayes":
            model = GaussianNB()
        elif model_option == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_option == "XGBoost":
            model = XGBClassifier(n_jobs=1, verbosity=0)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        input_data = np.array([[PM2_5, PM10, NO, SO2, O3, CO]])
        prediction_encoded = model.predict(input_data)
        prediction = label_encoder.inverse_transform(prediction_encoded)

        st.write("### Prediction Result")
        st.write(f"Predicted AQI Category: **{prediction[0]}**")
        st.write(f"Model Accuracy: **{accuracy:.2f}**")

# Classification Report Page
def classification_report_page(data, label_encoder):
    st.title("📊 Classification Report")
    st.write("### Model Performance Evaluation")
    model_option = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Naive Bayes", "Random Forest", "XGBoost"])

    if model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_option == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_option == "Naive Bayes":
        model = GaussianNB()
    elif model_option == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_option == "XGBoost":
        model = XGBClassifier(n_jobs=1, verbosity=0)

    X = data[["PM2.5", "PM10", "NO", "SO2", "O3", "CO"]]
    y = data["AQI_Bucket_Encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    tab1, tab2 = st.tabs(["Training Data", "Testing Data"])

    with tab1:
        st.header(f"Training Data Performance ({model_option})")
        y_train_pred = model.predict(X_train)
        report = classification_report(y_train, y_train_pred, target_names=label_encoder.classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_train, y_train_pred)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax)
        plt.title(f"Training Data Confusion Matrix ({model_option})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

    with tab2:
        st.header(f"Testing Data Performance ({model_option})")
        y_test_pred = model.predict(X_test)
        report = classification_report(y_test, y_test_pred, target_names=label_encoder.classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_test_pred)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax)
        plt.title(f"Testing Data Confusion Matrix ({model_option})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

# About Page
def about_page():
    st.title("📄 About the Project")
    st.write("""
    This project aims to:
    - Analyze air quality data from various cities.
    - Predict the AQI category using machine learning models.
    - Provide insights into the factors affecting air quality.
    - Help policymakers and individuals take informed actions to improve air quality.
    """)
    st.write("### Technologies Used")
    st.write("""
    - Python
    - Streamlit
    - Scikit-learn
    - Pandas, NumPy, Matplotlib, Seaborn
    - XGBoost
    - Yellowbrick
    """)
    st.write("### Team Members")
    st.markdown("""
    - **Talha Khalid**  
      [LinkedIn](https://www.linkedin.com/in/talha-khalid-189092272) | [Kaggle](https://www.kaggle.com/talhachoudary) | [GitHub](https://github.com/talha142)
    - **Subhan Shahid**  
      [LinkedIn](https://www.linkedin.com/in/msubhanshahid)
    """)

# Main Function
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "EDA", "Prediction", "Classification Report", "About"])

    uploaded_file = st.sidebar.file_uploader("Upload Your Dataset (CSV)", type=["csv"])

    data = load_data(uploaded_file)
    cleaned_data = clean_data(data)
    encoded_data, label_encoder = encode_target(cleaned_data)

    if page == "Home":
        home_page()
    elif page == "EDA":
        eda_page(cleaned_data)
    elif page == "Prediction":
        prediction_page(encoded_data, label_encoder)
    elif page == "Classification Report":
        classification_report_page(encoded_data, label_encoder)
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
