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
def load_data():
    """
    Load the dataset from the specified file path.
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    data = pd.read_csv("city_day.csv")  # Read CSV file into DataFrame
    return data

# Data Cleaning
def clean_data(data):
    """
    Clean the dataset by handling missing values, removing duplicates, and removing outliers.
    Args:
        data (pandas.DataFrame): The raw dataset to clean
    Returns:
        pandas.DataFrame: The cleaned dataset
    """
    # Fill missing values for numerical columns with the mean
    numerical_columns = data.select_dtypes(include=['number']).columns  # Identify numerical columns
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

    # Fill missing values for categorical columns with the mode
    categorical_columns = data.select_dtypes(include=['object']).columns  # Identify categorical columns
    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

    # Remove duplicate rows
    data.drop_duplicates(inplace=True)

    # Remove outliers using the Interquartile Range (IQR) method
    Q1 = data[numerical_columns].quantile(0.25)  # First quartile (25th percentile)
    Q3 = data[numerical_columns].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
    upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers

    # Filter out outliers for each numerical column
    for column in numerical_columns:
        data = data[(data[column] >= lower_bound[column]) & (data[column] <= upper_bound[column])]

    return data

# Encode Target Variable
def encode_target(data):
    """
    Encode the target variable (`AQI_Bucket`) into numeric labels using LabelEncoder.
    Args:
        data (pandas.DataFrame): The dataset containing the target variable
    Returns:
        tuple: (encoded_data, label_encoder) where:
            - encoded_data: DataFrame with encoded target column
            - label_encoder: The fitted LabelEncoder object for inverse transformations
    """
    label_encoder = LabelEncoder()  # Initialize label encoder
    data['AQI_Bucket_Encoded'] = label_encoder.fit_transform(data['AQI_Bucket'])  # Encode target column
    return data, label_encoder

# Home Page
def home_page():
    """
    Display the Home Page with information about AQI, its importance, causes, impacts, and more.
    This page serves as an introduction to the application and AQI concepts.
    """
    st.title("🌍 Air Quality Index (AQI) Prediction and Analysis")
    st.write("Welcome to the AQI Project! This application helps you understand, analyze, and predict air quality using machine learning models.")

    # What is AQI? section
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

    # Continue with other sections (Importance, Control Methods, Causes, Impacts, Ranges)
    # ... [rest of the home page content remains the same]

# Data Analytics Page
def data_analytics_page(data):
    """
    Display the Data Analytics Page with dataset overview, statistics, and visualizations.
    Args:
        data (pandas.DataFrame): The dataset to analyze and visualize
    """
    st.title("📊 Data Analytics")
    
    # Show first 10 rows of the dataset
    st.write("### Dataset Overview")
    st.write(data.head(10))

    # Display basic statistics
    st.write("### Basic Statistics")
    st.write(data.describe())

    # Correlation Heatmap visualization
    st.write("### Correlation Heatmap")
    corr_matrix = data[['PM2.5', 'PM10', 'NO', 'SO2', 'O3', 'CO', 'AQI']].corr()  # Calculate correlations
    fig, ax = plt.subplots(figsize=(8, 6))  # Create figure
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)  # Plot heatmap
    st.pyplot(fig)  # Display plot in Streamlit

    # Average AQI per category visualization
    st.write("### Average AQI per AQI Category")
    group_by = data.groupby("AQI_Bucket")["AQI"].mean()  # Group by AQI category and calculate mean
    fig, ax = plt.subplots(figsize=(10, 5))  # Create figure
    sns.barplot(x=group_by.index, y=group_by.values, palette="coolwarm", ax=ax)  # Create bar plot
    plt.xlabel("AQI Category")  # X-axis label
    plt.ylabel("Average AQI")  # Y-axis label
    plt.title("Average AQI per AQI Category")  # Plot title
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    st.pyplot(fig)  # Display plot in Streamlit

# Prediction Page
def prediction_page(data, label_encoder):
    """
    Display the Prediction Page where users can select a model, input data, and view predictions.
    Args:
        data (pandas.DataFrame): The dataset used for training models
        label_encoder (LabelEncoder): Encoder for the target variable
    """
    st.title("🔮 AQI Prediction")
    st.write("### Select Features for Prediction")

    # Create three columns for better layout of input fields
    col1, col2, col3 = st.columns(3)

    # Column 1: Pollutant level inputs
    with col1:
        st.write("#### Pollutant Levels")
        PM2_5 = st.number_input("PM2.5", value=data["PM2.5"].mean(), step=0.1)  # Input for PM2.5
        PM10 = st.number_input("PM10", value=data["PM10"].mean(), step=0.1)  # Input for PM10

    # Column 2: Pollutant level inputs
    with col2:
        st.write("#### Pollutant Levels")
        NO = st.number_input("NO", value=data["NO"].mean(), step=0.1)  # Input for NO
        SO2 = st.number_input("SO2", value=data["SO2"].mean(), step=0.1)  # Input for SO2

    # Column 3: Pollutant level inputs
    with col3:
        st.write("#### Pollutant Levels")
        O3 = st.number_input("O3", value=data["O3"].mean(), step=0.1)  # Input for O3
        CO = st.number_input("CO", value=data["CO"].mean(), step=0.1)  # Input for CO

    # Model selection dropdown
    st.write("### Select Model")
    model_option = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Naive Bayes", "Random Forest", "XGBoost"])

    # Prediction button
    if st.button("Predict"):
        # Prepare data for model training and prediction
        X = data[["PM2.5", "PM10", "NO", "SO2", "O3", "CO"]]  # Features
        y = data["AQI_Bucket_Encoded"]  # Target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data

        # Initialize selected model
        if model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)  # Logistic Regression with increased max iterations
        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier()  # Decision Tree classifier
        elif model_option == "Naive Bayes":
            model = GaussianNB()  # Gaussian Naive Bayes classifier
        elif model_option == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)  # Random Forest with 100 trees
        elif model_option == "XGBoost":
            model = XGBClassifier()  # XGBoost classifier

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on test set for evaluation
        y_pred = model.predict(X_test)

        # Calculate accuracy score
        accuracy = accuracy_score(y_test, y_pred)

        # Make prediction for user input
        input_data = np.array([[PM2_5, PM10, NO, SO2, O3, CO]])  # Prepare user input
        prediction_encoded = model.predict(input_data)  # Make prediction
        prediction = label_encoder.inverse_transform(prediction_encoded)  # Decode prediction to original label

        # Display results
        st.write("### Prediction Result")
        st.write(f"Predicted AQI Category: **{prediction[0]}**")  # Show predicted category
        st.write(f"Model Accuracy: **{accuracy:.2f}**")  # Show model accuracy

# Classification Report Page

        
            
    
        def classification_report_page(data, label_encoder):
    """
    Display the Classification Report Page with model performance metrics and visualizations.
    Allows selection of different models like the prediction page.
    """
    st.title("📊 Classification Report")
    st.write("### Model Performance Evaluation")

    # Model selection dropdown - same options as prediction page
    model_option = st.selectbox("Select Model", 
                              ["Logistic Regression", "Decision Tree", 
                               "Naive Bayes", "Random Forest", "XGBoost"])
    
    # Initialize selected model - same as prediction page
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

    # Prepare features and target
    X = data[["PM2.5", "PM10", "NO", "SO2", "O3", "CO"]]
    y = data["AQI_Bucket_Encoded"]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the selected model
    model.fit(X_train, y_train)

    # Create tabs for training and testing results
    tab1, tab2 = st.tabs(["Training Data", "Testing Data"])

    # Training Data Tab
    with tab1:
        st.header(f"Training Data Performance ({model_option})")
        
        # Make predictions on training data
        y_train_pred = model.predict(X_train)
        
        # Display classification report
        st.subheader("Classification Report")
        report = classification_report(
            y_train, 
            y_train_pred, 
            target_names=label_encoder.classes_,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
        
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_train, y_train_pred)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            cm, 
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax
        )
        plt.title(f"Training Data Confusion Matrix ({model_option})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

    # Testing Data Tab
    with tab2:
        st.header(f"Testing Data Performance ({model_option})")
        
        # Make predictions on testing data
        y_test_pred = model.predict(X_test)
        
        # Display classification report
        st.subheader("Classification Report")
        report = classification_report(
            y_test,
            y_test_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
        
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_test_pred)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax
        )
        plt.title(f"Testing Data Confusion Matrix ({model_option})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
# About Page
def about_page():
    """Display the About Page with project information and team details."""
    st.title("📄 About the Project")
    st.write("""
    This project aims to:
    - Analyze air quality data from various cities.
    - Predict the AQI category using machine learning models.
    - Provide insights into the factors affecting air quality.
    """)
    
    # ... [rest of the about page content remains the same]

# Main Function
def main():
    """
    Main function to run the Streamlit application.
    Handles navigation and page routing.
    """
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Analytics", "Prediction", "Classification Report", "About"])

    # Load and preprocess data
    data = load_data()  # Load raw data
    cleaned_data = clean_data(data)  # Clean data
    encoded_data, label_encoder = encode_target(cleaned_data)  # Encode target variable

    # Route to selected page
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

# Entry point
if __name__ == "__main__":
    main()  # Run the application
