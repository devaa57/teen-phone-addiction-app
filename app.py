import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ML Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

# ---------------------------
# Load & Preprocess Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("teen_phone_addiction_dataset.csv")

    # Bin Addiction Level → 3 classes
    bins = [0, 5, 8, 10]
    labels = ["Low", "Medium", "High"]
    df["Addiction_Class"] = pd.cut(df["Addiction_Level"], bins=bins, labels=labels, include_lowest=True)

    # Drop ID, Name, Addiction_Level for modeling
    df_model = df.drop(["ID", "Name", "Addiction_Level"], axis=1)

    # Encode categorical features
    cat_cols = [col for col in df_model.select_dtypes(include=["object", "category"]).columns 
                if col != "Addiction_Class"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        le_dict[col] = le

    X = df_model.drop("Addiction_Class", axis=1)
    y = df_model["Addiction_Class"]

    return X, y, df, df_model, cat_cols, le_dict

# ---------------------------
# EDA Section
# ---------------------------
def show_eda_section(df):
    st.header("📊 Exploratory Data Analysis (EDA)")

    # Dataset Overview
    st.subheader("Dataset Overview")
    st.write(df.head())

    # Basic info
    st.subheader("Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write("Data Types:")
    st.write(df.dtypes)

    # Summary statistics
    st.subheader("Statistical Summary")
    st.write(df.describe())

    # Numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Histogram
    st.subheader("1️⃣ Distribution of Daily Phone Usage Hours")
    fig, ax = plt.subplots()
    sns.histplot(df["Daily_Usage_Hours"], kde=True, ax=ax, color='skyblue')
    st.pyplot(fig)

    # Bar plot
    st.subheader("2️⃣ Average Addiction Level by Gender")
    fig, ax = plt.subplots()
    sns.barplot(x="Gender", y="Addiction_Level", data=df, ax=ax)
    st.pyplot(fig)

    # Scatter plot
    st.subheader("3️⃣ Relationship Between Daily Usage and Sleep Hours")
    fig, ax = plt.subplots()
    sns.scatterplot(x="Daily_Usage_Hours", y="Sleep_Hours", hue="Gender", data=df, ax=ax)
    st.pyplot(fig)

    # Box plot
    st.subheader("4️⃣ Addiction Level by School Grade")
    fig, ax = plt.subplots()
    sns.boxplot(x="School_Grade", y="Addiction_Level", data=df, ax=ax)
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("5️⃣ Correlation Heatmap")
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Insights
    st.markdown("### 🔍 Insights:")
    st.markdown("""
    - Teens with **higher daily usage hours** show higher addiction levels.  
    - **Lower sleep hours** correlate with higher phone usage.  
    - **School grade** influences addiction — higher grades may show more control.  
    - Strongest correlations can be found in the heatmap above.
    """)

# ---------------------------
# Main App
# ---------------------------
st.title("📱 Teen Phone Addiction App")

# Load data
X, y, df, df_model, cat_cols, le_dict = load_data()
classes = ["Low", "Medium", "High"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["EDA 📊", "ML Prediction 🔮", "Model Comparison 📈"])

# ---------------------------
# Tab 1: EDA
# ---------------------------
with tab1:
    show_eda_section(df)

# ---------------------------
# Tab 2: ML Prediction
# ---------------------------
with tab2:
    st.sidebar.header("Choose Model")
    model_name = st.sidebar.selectbox(
        "Select a model:",
        ["Decision Tree", "Logistic Regression", "kNN", "SVM", 
         "Random Forest", "Gradient Boosting", "Naive Bayes", 
         "Neural Network", "KMeans (unsupervised)"]
    )

    # Define model
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "kNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "SVM":
        model = SVC(kernel="rbf", probability=True, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Neural Network":
        model = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)

    # Prediction Form
    st.subheader("🔮 Make a Prediction")
    with st.form("prediction_form"):
        input_data = {}
        for col in X.columns:
            if col in cat_cols:
                le = le_dict[col]
                input_data[col] = st.selectbox(col, le.classes_)
            else:
                input_data[col] = st.number_input(col, value=float(X[col].mean()))
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([input_data])
        # Encode categorical features
        for col in cat_cols:
            le = le_dict[col]
            input_df[col] = le.transform(input_df[col])
        input_scaled = scaler.transform(input_df)

        if model_name != "KMeans (unsupervised)":
            model.fit(X_train_scaled, y_train)
            prediction = model.predict(input_scaled)[0]
            st.success(f"📌 Predicted Addiction Level: **{prediction}**")
        else:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(X_train_scaled)
            cluster = kmeans.predict(input_scaled)[0]
            st.info(f"📌 Assigned Cluster: **{cluster}**")

# ---------------------------
# Tab 3: Model Comparison
# ---------------------------
with tab3:
    st.header("📈 Compare All Models")

    # Define all supervised models (exclude KMeans)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)
    }

    metrics_list = []

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics_list.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1-Score": f1_score(y_test, y_pred, average="weighted")
        })

    metrics_df = pd.DataFrame(metrics_list).sort_values(by="Accuracy", ascending=False)
    st.subheader("📋 Metrics Table")
    st.dataframe(metrics_df)

    st.subheader("📊 Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x="Model", y="Accuracy", data=metrics_df, palette="viridis", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

    # Confusion matrices for each model
    st.subheader("🗂 Confusion Matrices")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        st.markdown(f"**{name}**")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} Confusion Matrix")
        st.pyplot(fig)
