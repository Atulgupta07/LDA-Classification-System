import pandas as pd

def predict_new_sample(models_without_lda, models_with_lda, scaler, lda_model, feature_names, input_data):
    """
    Predicts the class of a new sample using the trained models.
    """
    df_sample = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale features
    sample_scaled = scaler.transform(df_sample)
    df_scaled = pd.DataFrame(sample_scaled, columns=feature_names)
    
    # Apply LDA
    sample_lda = lda_model.transform(df_scaled)
    lda_cols = [f'LDA_Component_{i+1}' for i in range(sample_lda.shape[1])]
    df_lda = pd.DataFrame(sample_lda, columns=lda_cols)
    
    predictions = {
        "Without LDA": {},
        "With LDA": {}
    }
    
    for name, model in models_without_lda.items():
        predictions["Without LDA"][name] = model.predict(df_scaled)[0]
        
    for name, model in models_with_lda.items():
        predictions["With LDA"][name] = model.predict(df_lda)[0]
        
    return predictions
