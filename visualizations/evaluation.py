import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_confusion_matrix(cm, class_names=None, model_name=""):
    """
    Plots a confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig

def plot_model_comparison(metrics_dict):
    """
    Plots a bar chart comparing model accuracies.
    """
    df = pd.DataFrame(metrics_dict).T
    df_metrics = df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df_metrics.plot(kind='bar', ax=ax)
    ax.set_title("Model Comparison")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
