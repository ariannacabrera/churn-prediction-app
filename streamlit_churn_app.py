"""
Customer Churn Prediction App
Save this as: streamlit_app.py
Run with: streamlit run streamlit_app.py
"""

import io
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Churn Prediction Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: bold;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = None
if 'production_model' not in st.session_state:
    st.session_state.production_model = None
if 'feature_schema' not in st.session_state:
    st.session_state.feature_schema = None

# ==================== HELPER FUNCTIONS ====================

def normalize_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Normalize target to 0/1 format"""
    if target_col not in df.columns:
        return df
    
    s = df[target_col].copy()
    
    # If it's already numeric, just ensure it's 0/1
    if pd.api.types.is_numeric_dtype(s):
        df[target_col] = pd.to_numeric(s, errors="coerce").fillna(0).clip(0, 1).astype(int)
    # If it's object/string type
    elif s.dtype == "O":
        def to_bin(x):
            if x is None or (isinstance(x, float) and pd.isna(x)): 
                return 0
            xs = str(x).strip().lower()
            if xs in {"yes", "true", "1", "1.0", "y", "t"}: 
                return 1
            if xs in {"no", "false", "0", "0.0", "n", "f"}: 
                return 0
            return 0
        df[target_col] = s.map(to_bin).astype(int)
    else:
        df[target_col] = s.astype(int)
    
    return df

def clean_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Clean and prepare data"""
    df = df.copy()
    
    # Drop ID columns first
    id_cols = [c for c in df.columns if 'id' in c.lower()]
    if id_cols:
        df = df.drop(columns=id_cols)
    
    # Normalize target column
    df = normalize_target(df, target_col)
    
    # Drop rows with missing target values
    df = df.dropna(subset=[target_col])
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Verify we have both classes
    unique_classes = df[target_col].unique()
    if len(unique_classes) < 2:
        raise ValueError(f"Target column must have at least 2 classes. Found only: {unique_classes}")
    
    return df

def build_preprocessor(df: pd.DataFrame, target_col: str):
    """Build preprocessing pipeline"""
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]
    
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    
    schema = {
        "target": target_col,
        "numeric_features": num_cols,
        "categorical_features": cat_cols
    }
    
    return preprocessor, schema

def train_models(df: pd.DataFrame, target_col: str, test_size: float = 0.2):
    """Train all models and return results"""
    df = df.copy()
    
    # Ensure target is int type
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    
    # Check for class balance before proceeding
    class_counts = y.value_counts()
    st.info(f"📊 Class distribution: {class_counts.to_dict()}")
    
    if len(class_counts) < 2:
        st.error(f"❌ Only found class {y.unique()[0]} in the data. Cannot train models.")
        st.stop()
    
    preprocessor, schema = build_preprocessor(df, target_col)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    models_def = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, model) in enumerate(models_def.items()):
        status_text.text(f"Training {name}...")
        
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }
        results.append(metrics)
        
        trained_models[name] = {
            "pipeline": pipe,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
        
        progress_bar.progress((idx + 1) / len(models_def))
    
    status_text.empty()
    progress_bar.empty()
    
    results_df = pd.DataFrame(results).set_index("model").sort_values("f1", ascending=False)
    
    return results_df, trained_models, schema, X_test

# ==================== MAIN APP ====================

st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction Platform</h1>', unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.selectbox(
    "📍 Navigation",
    ["📤 Upload Data", "🧹 Data Cleaning", "🤖 Train Models", "📊 Model Comparison", "🔮 Make Predictions"]
)

# ==================== PAGE 1: UPLOAD DATA ====================
if page == "📤 Upload Data":
    st.header("📤 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with customer data",
        type=['csv'],
        help="Your dataset should include customer features and a churn column"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df_raw = df
        
        st.success(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📊 Column Information")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Non-Null Count": df.count().values,
            "Null Count": df.isnull().sum().values
        })
        st.dataframe(info_df, use_container_width=True)

# ==================== PAGE 2: DATA CLEANING ====================
elif page == "🧹 Data Cleaning":
    st.header("🧹 Data Cleaning & Preparation")
    
    if st.session_state.df_raw is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        df = st.session_state.df_raw.copy()
        
        st.subheader("⚙️ Cleaning Options")
        
        target_col = st.selectbox(
            "Select Target Column (Churn)",
            options=df.columns.tolist(),
            index=len(df.columns)-1 if 'churn' in df.columns else 0
        )
        
        if st.button("🔄 Clean Data", type="primary"):
            try:
                with st.spinner("Cleaning data..."):
                    # Show original target distribution
                    st.info(f"📊 Original '{target_col}' values: {df[target_col].value_counts().to_dict()}")
                    
                    df_clean = clean_data(df, target_col)
                    st.session_state.df_clean = df_clean
                    st.session_state.target_col = target_col
                    
                    # Show cleaned target distribution
                    st.info(f"📊 Cleaned '{target_col}' values: {df_clean[target_col].value_counts().to_dict()}")
                
                st.success("✅ Data cleaned successfully!")
            except ValueError as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure your target column has both churned (1) and non-churned (0) customers.")
                
                # Debug info
                with st.expander("🔍 Debug Information"):
                    st.write(f"Target column: {target_col}")
                    st.write(f"Unique values in target: {df[target_col].unique()}")
                    st.write(f"Value counts: {df[target_col].value_counts()}")
                    st.write(f"Data type: {df[target_col].dtype}")
                st.stop()
            
            try:
                with st.spinner("Cleaning data..."):
                    df_clean = clean_data(df, target_col)
                    st.session_state.df_clean = df_clean
                    st.session_state.target_col = target_col
                
                st.success("✅ Data cleaned successfully!")
            except ValueError as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure your target column has both churned (1) and non-churned (0) customers.")
                st.stop()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows Before", len(df))
                st.metric("Rows After", len(df_clean))
            with col2:
                st.metric("Rows Removed", len(df) - len(df_clean))
                st.metric("Columns", len(df_clean.columns))
            
            st.subheader("📊 Cleaned Data Preview")
            st.dataframe(df_clean.head(10), use_container_width=True)
            
            # Churn distribution
            if target_col in df_clean.columns:
                st.subheader("🎯 Target Distribution")
                churn_counts = df_clean[target_col].dropna().value_counts().sort_index()
                
                if len(churn_counts) > 0:
                    labels = ['No Churn' if i == 0 else 'Churn' for i in churn_counts.index]
                    fig = px.pie(
                        values=churn_counts.values,
                        names=labels,
                        title="Churn Distribution",
                        color_discrete_sequence=['#667eea', '#764ba2']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show counts
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("No Churn", churn_counts.get(0, 0))
                    with col2:
                        st.metric("Churn", churn_counts.get(1, 0))
                else:
                    st.warning("⚠️ No valid churn data found in target column")

# ==================== PAGE 3: TRAIN MODELS ====================
elif page == "🤖 Train Models":
    st.header("🤖 Train Machine Learning Models")
    
    if st.session_state.df_clean is None:
        st.warning("⚠️ Please clean your data first!")
    else:
        df_clean = st.session_state.df_clean
        target_col = st.session_state.target_col
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        with col2:
            st.metric("Training Samples", int(len(df_clean) * (1 - test_size)))
        
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training models... This may take a minute."):
                results, models, schema, X_test = train_models(df_clean, target_col, test_size)
                
                st.session_state.results = results
                st.session_state.models = models
                st.session_state.feature_schema = schema
                st.session_state.X_test = X_test
            
            st.success("✅ All models trained successfully!")
            st.balloons()
            
            st.subheader("📊 Training Results")
            st.dataframe(results.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

# ==================== PAGE 4: MODEL COMPARISON ====================
elif page == "📊 Model Comparison":
    st.header("📊 Model Performance Comparison")
    
    if st.session_state.results is None:
        st.warning("⚠️ Please train models first!")
    else:
        results = st.session_state.results
        models = st.session_state.models
        
        # Metrics comparison
        st.subheader("📈 Performance Metrics")
        
        fig = go.Figure()
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            fig.add_trace(go.Bar(
                name=metric.upper(),
                x=results.index,
                y=results[metric],
                text=results[metric].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            barmode='group',
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        best_model = results.index[0]
        st.subheader("🏆 Best Model")
        st.success(f"**{best_model}** with F1 Score: {results.loc[best_model, 'f1']:.4f}")
        
        # Confusion matrices
        st.subheader("🔢 Confusion Matrices")
        cols = st.columns(3)
        
        for idx, (name, model_data) in enumerate(models.items()):
            with cols[idx]:
                cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['No Churn', 'Churn'],
                    y=['No Churn', 'Churn'],
                    title=name,
                    color_continuous_scale='Purples'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Deploy to production
        st.subheader("🚀 Deploy Model")
        selected_model = st.selectbox("Select model for production", results.index.tolist())
        
        if st.button("Deploy to Production", type="primary"):
            st.session_state.production_model = models[selected_model]['pipeline']
            st.success(f"✅ {selected_model} deployed to production!")

# ==================== PAGE 5: MAKE PREDICTIONS ====================
elif page == "🔮 Make Predictions":
    st.header("🔮 Predict Customer Churn")
    
    if st.session_state.production_model is None:
        st.warning("⚠️ Please deploy a model first!")
    else:
        pipe = st.session_state.production_model
        schema = st.session_state.feature_schema
        
        st.subheader("📝 Enter Customer Information")
        
        input_data = {}
        
        # Create input fields based on schema
        num_cols = schema['numeric_features']
        cat_cols = schema['categorical_features']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numeric Features**")
            for col in num_cols:
                input_data[col] = st.number_input(f"{col}", value=0.0)
        
        with col2:
            st.markdown("**Categorical Features**")
            for col in cat_cols:
                # Get unique values from cleaned data if available
                if st.session_state.df_clean is not None and col in st.session_state.df_clean.columns:
                    options = st.session_state.df_clean[col].unique().tolist()
                    input_data[col] = st.selectbox(f"{col}", options)
                else:
                    input_data[col] = st.text_input(f"{col}")
        
        if st.button("🎯 Predict Churn", type="primary"):
            # Create DataFrame with correct column order
            X = pd.DataFrame([input_data], columns=num_cols + cat_cols)
            
            prediction = pipe.predict(X)[0]
            proba = pipe.predict_proba(X)[0, 1]
            
            st.markdown("---")
            st.subheader("🔮 Prediction Result")
            
            if prediction == 1:
                st.error("⚠️ **Customer will CHURN**")
                st.markdown(f"### Churn Probability: {proba*100:.1f}%")
            else:
                st.success("✅ **Customer will NOT churn**")
                st.markdown(f"### Retention Probability: {(1-proba)*100:.1f}%")
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={'text': "Churn Risk"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if proba > 0.5 else "darkgreen"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgreen"},
                        {'range': [33, 66], 'color': "yellow"},
                        {'range': [66, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 App Status")
if st.session_state.df_raw is not None:
    st.sidebar.success("✅ Data Uploaded")
else:
    st.sidebar.info("⏳ Awaiting Data")

if st.session_state.df_clean is not None:
    st.sidebar.success("✅ Data Cleaned")
else:
    st.sidebar.info("⏳ Awaiting Cleaning")

if st.session_state.results is not None:
    st.sidebar.success("✅ Models Trained")
else:
    st.sidebar.info("⏳ Awaiting Training")

if st.session_state.production_model is not None:
    st.sidebar.success("✅ Model Deployed")
else:
    st.sidebar.info("⏳ Awaiting Deployment")
