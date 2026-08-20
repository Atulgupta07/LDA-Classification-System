import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_dataset(path=None):
    if path and os.path.exists(path):
        return pd.read_csv(path)
    
    # Fallback to loading Iris directly if not found
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Optionally save it for future
    os.makedirs(os.path.dirname(path) if path else 'data', exist_ok=True)
    save_path = path if path else 'data/iris.csv'
    df.to_csv(save_path, index=False)
    
    return df

def dataset_summary(df, target_column):
    return {
        "n_samples": df.shape[0],
        "n_features": df.shape[1] - 1,
        "n_classes": df[target_column].nunique(),
        "missing_values": df.isnull().sum().sum(),
        "class_counts": df[target_column].value_counts().to_dict()
    }

def preprocess(df, target_column, test_size=0.2, random_state=42):
    df = df.dropna()
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame/Series for easier handling later
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    df = load_dataset('data/iris.csv')
    summary = dataset_summary(df, 'target')
    print("Dataset Summary:", summary)
    X_train, X_test, y_train, y_test, scaler = preprocess(df, 'target')
    print("Preprocessing complete. X_train shape:", X_train.shape)
