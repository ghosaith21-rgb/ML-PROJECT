import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------- Page config ----------
st.set_page_config(page_title="Wine Quality Predictor", layout="wide")
st.title("🍷 Wine Quality Prediction using Naive Bayes")

# ---------- Load and prepare data (cached) ----------
@st.cache_data
def load_and_prepare():
    df = pd.read_csv("final_alcohol.csv")
    # Drop meaningless index columns
    df = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"], errors="ignore")
    # Clean and encode wine type
    df["Type"] = df["Type"].str.strip().replace({"White Wine": 1, "Red Wine": 0})
    return df

df = load_and_prepare()

# ---------- Train model (cached) ----------
@st.cache_resource
def train_model(data):
    X = data.drop("quality", axis=1)
    y = data["quality"]
    model = GaussianNB()
    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_names = train_model(df)

# ---------- Compute test accuracy (optional) ----------
# We'll do a quick train/test split to show model performance
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("quality", axis=1), df["quality"], test_size=0.3, random_state=20
)
test_acc = accuracy_score(y_test, model.predict(X_test))

# ---------- Sidebar: Input features ----------
st.sidebar.header("📊 Input Wine Features")
inputs = {}

for col in feature_names:
    if col == "Type":
        # Wine type selector
        type_choice = st.sidebar.selectbox("Wine Type", options=["White Wine", "Red Wine"])
        inputs[col] = 1 if type_choice == "White Wine" else 0
    else:
        # Numeric inputs – use sensible defaults from dataset mean
        default_val = float(df[col].mean())
        step = 0.01 if col not in ["alcohol"] else 0.1
        inputs[col] = st.sidebar.number_input(
            col.replace("_", " ").title(),
            value=default_val,
            step=step,
            format="%.3f" if "density" in col else "%.2f"
        )

# ---------- Prediction button ----------
if st.sidebar.button("🔮 Predict Quality"):
    input_df = pd.DataFrame([inputs])
    pred = model.predict(input_df)[0]
    st.sidebar.success(f"### Predicted Quality: **{pred}**")

# ---------- Main panel: show dataset overview & model info ----------
st.subheader("📈 Model Performance")
st.write(f"**Test Accuracy (30% hold‑out):** {test_acc:.4f}")
st.write("**Features used:**")
st.write(", ".join(feature_names))

st.subheader("📋 Raw Data Sample")
st.dataframe(df.head(10))

st.markdown("---")
st.caption("Model trained with Gaussian Naive Bayes on the combined wine quality dataset.")
