import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Wine Quality Analysis - Naive Bayes Classifier",
    page_icon="🍷",
    layout="wide"
)

# Title and description
st.title("🍷 Wine Quality Analysis using Naive Bayes Classifier")
st.markdown("""
This application analyzes wine quality using different Naive Bayes classifier algorithms.
The dataset contains various chemical properties of red and white wines.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("final_alcohol.csv")
    
    def clean_data(st):
        st = st.strip()
        return st
    
    df['Type'] = df['Type'].apply(clean_data)
    df['Type'] = df['Type'].replace(['White Wine', 'Red Wine'], [1, 0])
    return df

try:
    df = load_data()
    st.success("✅ Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please make sure 'final_alcohol.csv' is in the same directory as app.py")
    st.stop()

# Sidebar
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Data Overview", "Exploratory Analysis", "Model Performance", "Predict Wine Quality"]
)

# Data Overview Page
if page == "Data Overview":
    st.header("📋 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Data Sample")
        st.dataframe(df.head(10))
        
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    with col2:
        st.subheader("Data Information")
        buffer = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Unique Values': df.nunique().values
        })
        st.dataframe(buffer)
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
    
    st.subheader("Quality Distribution")
    quality_counts = df['quality'].value_counts().sort_index()
    fig = px.bar(x=quality_counts.index, y=quality_counts.values, 
                 labels={'x': 'Quality', 'y': 'Count'},
                 title='Distribution of Wine Quality Scores')
    st.plotly_chart(fig, use_container_width=True)

# Exploratory Analysis Page
elif page == "Exploratory Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    # Select columns for analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_cols = st.multiselect("Select columns to analyze", numeric_cols, default=['alcohol', 'quality', 'Type'])
    
    if selected_cols:
        # Box plots
        st.subheader("Box Plots")
        for col in selected_cols:
            if col != 'quality':  # Skip quality for box plot
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.boxplot(data=df, x='quality', y=col, ax=ax)
                plt.title(f'{col} Distribution by Quality')
                st.pyplot(fig)
                plt.close()
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(df.select_dtypes(include=['int64', 'float64']).corr(), 
                annot=True, cmap='coolwarm', center=0, ax=ax)
    plt.title('Feature Correlation Matrix')
    st.pyplot(fig)
    plt.close()
    
    # Quality count plot
    st.subheader("Wine Quality Distribution")
    fig = px.histogram(df, x='quality', color='Type', 
                       title='Quality Distribution by Wine Type',
                       labels={'Type': 'Wine Type (1=White, 0=Red)'})
    st.plotly_chart(fig, use_container_width=True)

# Model Performance Page
elif page == "Model Performance":
    st.header("🤖 Naive Bayes Classifier Performance")
    
    # Prepare data
    x = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values
    
    # Split data
    test_size = st.slider("Test Size Ratio", 0.1, 0.4, 0.3, 0.05)
    random_state = st.number_input("Random State", 0, 100, 20)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    st.write(f"Training set size: {x_train.shape[0]} samples")
    st.write(f"Test set size: {x_test.shape[0]} samples")
    
    # Train and evaluate models
    models = {
        'Gaussian Naive Bayes': GaussianNB(),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Bernoulli Naive Bayes': BernoulliNB()
    }
    
    results = []
    predictions = {}
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (name, model) in enumerate(models.items()):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        predictions[name] = y_pred
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
        
        # Display metrics in columns
        with [col1, col2, col3][idx]:
            st.subheader(name)
            st.metric("Accuracy", f"{accuracy:.3f}")
            st.metric("Precision", f"{precision:.3f}")
            st.metric("Recall", f"{recall:.3f}")
            st.metric("F1 Score", f"{f1:.3f}")
    
    # Results comparison
    st.subheader("Model Comparison")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)
    
    # Bar chart comparison
    fig = go.Figure()
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        fig.add_trace(go.Bar(
            name=metric,
            x=results_df['Model'],
            y=results_df[metric],
            text=results_df[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Model Performance Metrics Comparison",
        barmode='group',
        yaxis_range=[0, 1],
        yaxis_title="Score"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrix for selected model
    st.subheader("Confusion Matrix")
    selected_model = st.selectbox("Select model for confusion matrix", list(models.keys()))
    
    cm = confusion_matrix(y_test, predictions[selected_model])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title(f'Confusion Matrix - {selected_model}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)
    plt.close()

# Predict Wine Quality Page
elif page == "Predict Wine Quality":
    st.header("🔮 Predict Wine Quality")
    
    st.markdown("""
    Enter the wine characteristics below to predict its quality using the Gaussian Naive Bayes model.
    """)
    
    # Get feature names (excluding target)
    feature_names = df.columns[:-1].tolist()
    
    # Create input fields for each feature
    col1, col2 = st.columns(2)
    
    input_data = []
    with col1:
        for i, feature in enumerate(feature_names[:len(feature_names)//2]):
            value = st.number_input(
                f"{feature}",
                value=float(df[feature].mean()),
                step=0.1,
                format="%.2f"
            )
            input_data.append(value)
    
    with col2:
        for i, feature in enumerate(feature_names[len(feature_names)//2:]):
            value = st.number_input(
                f"{feature}",
                value=float(df[feature].mean()),
                step=0.1,
                format="%.2f"
            )
            input_data.append(value)
    
    # Train model for prediction
    x = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values
    
    model = GaussianNB()
    model.fit(x, y)
    
    if st.button("Predict Quality", type="primary"):
        # Make prediction
        prediction = model.predict([input_data])[0]
        prediction_proba = model.predict_proba([input_data])[0]
        
        st.subheader("Prediction Result")
        
        # Display prediction
        col1, col2, col3 = st.columns(3)
        with col2:
            st.metric("Predicted Quality", f"{prediction:.0f}")
        
        # Display probability distribution
        st.subheader("Probability Distribution")
        proba_df = pd.DataFrame({
            'Quality': model.classes_,
            'Probability': prediction_proba
        })
        
        fig = px.bar(proba_df, x='Quality', y='Probability', 
                     title='Probability Distribution for Each Quality Score',
                     labels={'Probability': 'Probability', 'Quality': 'Quality Score'})
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This app demonstrates wine quality classification using Naive Bayes algorithms.
- **Dataset**: Wine Quality Dataset
- **Models**: Gaussian, Multinomial, Bernoulli NB
- **Features**: Chemical properties of wines
""")

# Run the app
if __name__ == '__main__':
    st.sidebar.markdown("---")
    st.sidebar.info("👆 Use the navigation menu to explore different sections")
