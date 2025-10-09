import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
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
@st.cache_data
def load_data():
    df = pd.read_csv("teen_phone_addiction_dataset.csv")

    # Bin Addiction Level → 3 classes
    bins = [0, 5, 8, 10]
    labels = ["Low", "Medium", "High"]
    df["Addiction_Class"] = pd.cut(df["Addiction_Level"], bins=bins, labels=labels, include_lowest=True)

    # Drop ID, Name, Addiction_Level
    df = df.drop(["ID", "Name", "Addiction_Level"], axis=1)

    # Encode categorical features
    # include object or category types, but exclude the target
    cat_cols = [col for col in df.select_dtypes(include=["object", "category"]).columns 
                if col != "Addiction_Class"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le  # save encoders for future use

    X = df.drop("Addiction_Class", axis=1)
    y = df["Addiction_Class"]

    return X, y, df, cat_cols, le_dict


# ---------------------------
# Main App
# ---------------------------
st.title("📱 Teen Phone Addiction Classification")

# Load data
X, y, df, cat_cols, le_dict = load_data()
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
# Sidebar: Model Selection
# ---------------------------
st.sidebar.header("Choose Model")
model_name = st.sidebar.selectbox(
    "Select a model:",
    ["Decision Tree", "Logistic Regression", "kNN", "SVM", 
     "Random Forest", "Gradient Boosting", "Naive Bayes", 
     "Neural Network", "KMeans (unsupervised)"]
)

# ---------------------------
# Define Models
# ---------------------------
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
    model = MLPClassifier(hidden_layer_sizes=(100,50), activation="relu",
                          solver="adam", max_iter=500, random_state=42)

# ---------------------------
# 🔮 Prediction Section
# ---------------------------
st.subheader("🔮 Make a Prediction")

with st.form("prediction_form"):
    input_data = {}
    for col in X.columns:
        if col in cat_cols:
            # Use original labels for selectbox
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

    # Scale numeric features
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
# 📊 Model Evaluation
# ---------------------------
st.subheader(f"📊 {model_name} Evaluation on Dataset")

if model_name != "KMeans (unsupervised)":
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

else:
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    y_pred_clusters = kmeans.predict(X_test_scaled)

    ari = adjusted_rand_score(y_test, y_pred_clusters)
    sil = silhouette_score(X_test_scaled, y_pred_clusters)
    st.write(f"**Adjusted Rand Index (vs true labels):** {ari:.3f}")
    st.write(f"**Silhouette Score:** {sil:.3f}")
