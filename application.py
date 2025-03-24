import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Import Yellowbrick after installation
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix
import yellowbrick

st.write(f'Yellowbrick version: {yellowbrick.__version__}')





# Load Dataset
@st.cache_data  # Cache the data to improve performance
def load_data():
    """
    Load the dataset from the specified file path.
    """
    data = pd.read_csv("city_day.csv")
    return data

# Data Cleaning
def clean_data(data):
    """
    Clean the dataset by handling missing values, removing duplicates, and removing outliers.
    """
    # Fill missing values for numerical columns with the mean
    numerical_columns = data.select_dtypes(include=['number']).columns
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

    # Fill missing values for categorical columns with the mode
    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

    # Remove duplicates
    data.drop_duplicates(inplace=True)

    # Remove outliers using the Interquartile Range (IQR) method
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
    """
    Encode the target variable (`AQI_Bucket`) into numeric labels using LabelEncoder.
    """
    label_encoder = LabelEncoder()
    data['AQI_Bucket_Encoded'] = label_encoder.fit_transform(data['AQI_Bucket'])
    return data, label_encoder

# Home Page
def home_page():
    """
    Display the Home Page with information about AQI, its importance, causes, impacts, and more.
    """
    st.title("🌍 Air Quality Index (AQI) Prediction and Analysis")
    st.write("Welcome to the AQI Project! This application helps you understand, analyze, and predict air quality using machine learning models.")

    # What is AQI?
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
    
    The AQI value ranges from **0 to 500**, with higher values indicating worse air quality.
    """)

    # Importance of AQI
    st.header("🌟 Why is AQI Important?")
    st.write("""
    Monitoring AQI is crucial for several reasons:
    - **Health Protection:** Poor air quality can cause respiratory and cardiovascular diseases, especially in vulnerable populations like children, the elderly, and those with pre-existing conditions.
    - **Environmental Impact:** Air pollution harms ecosystems, contributes to climate change, and reduces visibility.
    - **Policy Making:** Governments and organizations use AQI data to implement pollution control measures and improve public health.
    """)

    # How to Control AQI
    st.header("🛠️ How to Control AQI?")
    st.write("""
    Controlling air pollution requires collective efforts from individuals, industries, and governments. Here are some ways to improve AQI:
    - **Reduce Vehicle Emissions:** Use public transportation, carpool, or switch to electric vehicles.
    - **Promote Renewable Energy:** Shift from fossil fuels to solar, wind, and other renewable energy sources.
    - **Plant Trees:** Trees absorb CO2 and other pollutants, improving air quality.
    - **Regulate Industries:** Enforce strict emission standards for factories and power plants.
    - **Public Awareness:** Educate people about the harmful effects of air pollution and encourage sustainable practices.
    """)

    # Causes of Poor AQI
    st.header("🚨 Causes of Poor AQI")
    st.write("""
    Poor air quality is caused by a combination of natural and human-made factors:
    - **Vehicle Emissions:** Cars, trucks, and buses release pollutants like NOx, CO, and PM2.5.
    - **Industrial Activities:** Factories and power plants emit SO2, NOx, and particulate matter.
    - **Construction and Dust:** Construction sites and unpaved roads generate PM10 and PM2.5.
    - **Wildfires:** Burning forests release large amounts of smoke and particulate matter.
    - **Agricultural Activities:** Burning crop residues and using fertilizers release ammonia and other pollutants.
    - **Natural Sources:** Dust storms, volcanic eruptions, and pollen can also contribute to poor air quality.
    """)

    # Impacts of Poor AQI
    st.header("💔 Impacts of Poor AQI")
    st.write("""
    Poor air quality has severe consequences for health, the environment, and the economy:
    - **Health Impacts:**
        - Respiratory diseases (e.g., asthma, bronchitis)
        - Cardiovascular diseases (e.g., heart attacks, strokes)
        - Reduced lung function and development in children
        - Increased risk of lung cancer
    - **Environmental Impacts:**
        - Acid rain, which damages soil, water bodies, and vegetation
        - Global warming and climate change due to greenhouse gases
        - Reduced visibility (haze) affecting transportation and tourism
    - **Economic Impacts:**
        - Increased healthcare costs due to air pollution-related illnesses
        - Loss of productivity from sick workers
        - Damage to crops and ecosystems
    """)

    # AQI Ranges and Categories
    st.header("📊 AQI Ranges and Categories")
    st.write("""
    The AQI is divided into six categories, each corresponding to a different level of health concern:
    """)
    st.markdown("""
    | AQI Range | Category          | Health Implications                                                                 |
    |-----------|-------------------|-------------------------------------------------------------------------------------|
    | 0-50      | **Good**          | Air quality is satisfactory, and air pollution poses little or no risk.             |
    | 51-100    | **Moderate**      | Air quality is acceptable; however, some pollutants may be a concern for sensitive groups. |
    | 101-150   | **Unhealthy for Sensitive Groups** | Sensitive individuals may experience health effects; the general public is less likely to be affected. |
    | 151-200   | **Unhealthy**     | Everyone may begin to experience health effects; sensitive groups may experience more serious effects. |
    | 201-300   | **Very Unhealthy**| Health warnings of emergency conditions; the entire population is more likely to be affected. |
    | 301+      | **Hazardous**     | Health alert: everyone may experience serious health effects.                       |
    """)
    st.write("""
    **Note:** Sensitive groups include children, the elderly, and individuals with respiratory or cardiovascular diseases.
    """)

# Data Analytics Page
def data_analytics_page(data):
    """
    Display the Data Analytics Page with dataset overview, statistics, and visualizations.
    """
    st.title("📊 Data Analytics")
    st.write("### Dataset Overview")
    st.write(data.head(10))

    st.write("### Basic Statistics")
    st.write(data.describe())

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    corr_matrix = data[['PM2.5', 'PM10', 'NO', 'SO2', 'O3', 'CO', 'AQI']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Average AQI per AQI Category
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
    """
    Display the Prediction Page where users can select a model, input data, and view predictions.
    """
    st.title("🔮 AQI Prediction")
    st.write("### Select Features for Prediction")

    # Split the page into 3 columns for user input
    col1, col2, col3 = st.columns(3)

    # Column 1: User Input
    with col1:
        st.write("#### Pollutant Levels")
        PM2_5 = st.number_input("PM2.5", value=data["PM2.5"].mean(), step=0.1)
        PM10 = st.number_input("PM10", value=data["PM10"].mean(), step=0.1)

    # Column 2: User Input
    with col2:
        st.write("#### Pollutant Levels")
        NO = st.number_input("NO", value=data["NO"].mean(), step=0.1)
        SO2 = st.number_input("SO2", value=data["SO2"].mean(), step=0.1)

    # Column 3: User Input
    with col3:
        st.write("#### Pollutant Levels")
        O3 = st.number_input("O3", value=data["O3"].mean(), step=0.1)
        CO = st.number_input("CO", value=data["CO"].mean(), step=0.1)

    # Model Selection
    st.write("### Select Model")
    model_option = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Naive Bayes", "Random Forest", "XGBoost"])

    # Predict Button
    if st.button("Predict"):
        # Prepare data for prediction
        X = data[["PM2.5", "PM10", "NO", "SO2", "O3", "CO"]]
        y = data["AQI_Bucket_Encoded"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train and predict based on selected model
        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_option == "Naive Bayes":
            model = GaussianNB()
        elif model_option == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_option == "XGBoost":
            model = XGBClassifier()

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Make prediction for user input
        input_data = np.array([[PM2_5, PM10, NO, SO2, O3, CO]])
        prediction_encoded = model.predict(input_data)
        prediction = label_encoder.inverse_transform(prediction_encoded)  # Decode the prediction

        # Display prediction result and accuracy
        st.write("### Prediction Result")
        st.write(f"Predicted AQI Category: **{prediction[0]}**")
        st.write(f"Model Accuracy: **{accuracy:.2f}**")

# Classification Report Page
def classification_report_page(data, label_encoder):
    """
    Display the Classification Report Page with visualizations for both training and testing data.
    """
    st.title("📊 Classification Report")
    st.write("### Confusion Matrix and Classification Report for Training and Testing Data")

    # Prepare data for classification
    X = data[["PM2.5", "PM10", "NO", "SO2", "O3", "CO"]]
    y = data["AQI_Bucket_Encoded"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier (you can change the model if needed)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on Training Data
    st.write("#### Training Data Evaluation")

    # Classification Report for Training Data
    st.write("##### Classification Report (Training Data)")
    fig, ax = plt.subplots(figsize=(8, 6))
    visualizer_train = ClassificationReport(model, classes=label_encoder.classes_, support=True, ax=ax)
    visualizer_train.fit(X_train, y_train)
    visualizer_train.score(X_train, y_train)
    visualizer_train.show()
    st.pyplot(fig)

    # Confusion Matrix for Training Data
    st.write("##### Confusion Matrix (Training Data)")
    fig, ax = plt.subplots(figsize=(8, 6))
    visualizer_train_cm = ConfusionMatrix(model, classes=label_encoder.classes_, ax=ax)
    visualizer_train_cm.fit(X_train, y_train)
    visualizer_train_cm.score(X_train, y_train)
    visualizer_train_cm.show()
    st.pyplot(fig)

    # Evaluate on Testing Data
    st.write("#### Testing Data Evaluation")

    # Classification Report for Testing Data
    st.write("##### Classification Report (Testing Data)")
    fig, ax = plt.subplots(figsize=(8, 6))
    visualizer_test = ClassificationReport(model, classes=label_encoder.classes_, support=True, ax=ax)
    visualizer_test.fit(X_train, y_train)  # Fit on training data
    visualizer_test.score(X_test, y_test)  # Score on testing data
    visualizer_test.show()
    st.pyplot(fig)

    # Confusion Matrix for Testing Data
    st.write("##### Confusion Matrix (Testing Data)")
    fig, ax = plt.subplots(figsize=(8, 6))
    visualizer_test_cm = ConfusionMatrix(model, classes=label_encoder.classes_, ax=ax)
    visualizer_test_cm.fit(X_train, y_train)  # Fit on training data
    visualizer_test_cm.score(X_test, y_test)  # Score on testing data
    visualizer_test_cm.show()
    st.pyplot(fig)

# About Page
def about_page():
    """
    Display the About Page with project scope, technologies used, and team member details.
    """
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
    st.write("""
    - **Talha Khalid**
      - LinkedIn: [linkedin.com/in/talha-khalid-189092272](https://www.linkedin.com/in/talha-khalid-189092272)
      - Kaggle: [kaggle.com/talhachoudary](https://www.kaggle.com/talhachoudary)
      - GitHub: [github.com/talha142](https://github.com/talha142)
    - **Subhan Shahid**
      - LinkedIn: [linkedin.com/in/msubhanshahid](https://www.linkedin.com/in/msubhanshahid/)
    """)

# Main Function
def main():
    """
    Main function to run the Streamlit application.
    """
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Analytics", "Prediction", "Classification Report", "About"])

    data = load_data()
    cleaned_data = clean_data(data)
    encoded_data, label_encoder = encode_target(cleaned_data)  # Encode the target variable

    if page == "Home":
        home_page()
    elif page == "Data Analytics":
        data_analytics_page(cleaned_data)
    elif page == "Prediction":
        prediction_page(encoded_data, label_encoder)
    elif page == "Classification Report":
        classification_report_page(encoded_data, label_encoder)
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
