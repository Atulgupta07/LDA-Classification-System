import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from src.preprocessing import load_dataset, dataset_summary, preprocess
from src.lda import apply_lda
from visualizations.before_lda import plot_raw_data
from visualizations.after_lda import plot_lda_data
from src.classification import train_models
from src.evaluation import evaluate_models
from visualizations.evaluation import plot_confusion_matrix, plot_model_comparison
from src.prediction import predict_new_sample

st.set_page_config(page_title="LDA Classification System", layout="wide")

st.title("LDA Classification System")
st.markdown("Learn how Linear Discriminant Analysis (LDA) improves class separability and classification performance.")

# --- Session State ---
if 'df' not in st.session_state:
    st.session_state.df = load_dataset()
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False
if 'lda_done' not in st.session_state:
    st.session_state.lda_done = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

tabs = st.tabs(["1. Dataset", "2. Preprocessing", "3. LDA & Visualizations", "4. Classification & Evaluation", "5. Prediction"])

df = st.session_state.df

with tabs[0]:
    st.header("Raw Dataset")
    st.write(df.head())
    
    st.subheader("Dataset Summary")
    summary = dataset_summary(df, 'target')
    st.json(summary)
    
with tabs[1]:
    st.header("Preprocessing")
    st.write("We will perform a train/test split and scale the features.")
    
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    if st.button("Run Preprocessing"):
        X_train, X_test, y_train, y_test, scaler = preprocess(df, 'target', test_size=test_size)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.scaler = scaler
        st.session_state.preprocessing_done = True
        st.success("Preprocessing completed!")
        
    if st.session_state.preprocessing_done:
        st.write("X_train shape:", st.session_state.X_train.shape)
        st.write("X_test shape:", st.session_state.X_test.shape)

with tabs[2]:
    st.header("Linear Discriminant Analysis (LDA)")
    if st.session_state.preprocessing_done:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Before LDA")
            fig_before = plot_raw_data(df, 'target')
            st.pyplot(fig_before)
            
        with col2:
            st.subheader("After LDA")
            n_comp = st.selectbox("Number of LDA Components", [1, 2])
            if st.button("Apply LDA"):
                lda_model, X_train_lda, X_test_lda = apply_lda(
                    st.session_state.X_train, st.session_state.y_train, 
                    st.session_state.X_test, n_components=n_comp
                )
                st.session_state.lda_model = lda_model
                st.session_state.X_train_lda = X_train_lda
                st.session_state.X_test_lda = X_test_lda
                st.session_state.lda_done = True
                st.success("LDA Applied!")
                
            if st.session_state.lda_done:
                fig_after = plot_lda_data(st.session_state.X_train_lda, st.session_state.y_train)
                st.pyplot(fig_after)
    else:
        st.warning("Please complete preprocessing first.")

with tabs[3]:
    st.header("Classification & Evaluation")
    if st.session_state.lda_done:
        if st.button("Train Models & Evaluate"):
            # Train Without LDA
            models_no_lda = train_models(st.session_state.X_train, st.session_state.y_train)
            res_no_lda = evaluate_models(models_no_lda, st.session_state.X_test, st.session_state.y_test)
            
            # Train With LDA
            models_lda = train_models(st.session_state.X_train_lda, st.session_state.y_train)
            res_lda = evaluate_models(models_lda, st.session_state.X_test_lda, st.session_state.y_test)
            
            st.session_state.models_no_lda = models_no_lda
            st.session_state.models_lda = models_lda
            st.session_state.res_no_lda = res_no_lda
            st.session_state.res_lda = res_lda
            st.session_state.models_trained = True
            
        if st.session_state.models_trained:
            st.subheader("Comparison")
            st.write("Without LDA Metrics:")
            fig_comp_no_lda = plot_model_comparison(st.session_state.res_no_lda)
            st.pyplot(fig_comp_no_lda)
            
            st.write("With LDA Metrics:")
            fig_comp_lda = plot_model_comparison(st.session_state.res_lda)
            st.pyplot(fig_comp_lda)
            
            st.subheader("Confusion Matrices (With LDA)")
            cols = st.columns(3)
            for i, (name, metrics) in enumerate(st.session_state.res_lda.items()):
                with cols[i]:
                    fig_cm = plot_confusion_matrix(metrics['Confusion Matrix'], model_name=name)
                    st.pyplot(fig_cm)
    else:
        st.warning("Please apply LDA first.")

with tabs[4]:
    st.header("Prediction System")
    if st.session_state.models_trained:
        st.write("Enter feature values for a new sample:")
        feature_names = df.drop(columns=['target']).columns.tolist()
        
        input_data = []
        cols = st.columns(len(feature_names))
        for i, f_name in enumerate(feature_names):
            val = cols[i].number_input(f_name, value=float(df[f_name].mean()))
            input_data.append(val)
            
        if st.button("Predict"):
            preds = predict_new_sample(
                st.session_state.models_no_lda, 
                st.session_state.models_lda, 
                st.session_state.scaler, 
                st.session_state.lda_model, 
                feature_names, 
                input_data
            )
            st.subheader("Predictions")
            st.json(preds)
    else:
        st.warning("Please train models first.")
