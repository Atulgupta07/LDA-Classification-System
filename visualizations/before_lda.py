import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd

def plot_raw_data(df, target_column):
    """
    Plots the first two features or a 2D PCA of the raw dataset.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plot_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    else:
        plot_df = X.copy()
        plot_df.columns = ['Feature 1', 'Feature 2']
        
    plot_df['Target'] = y.values
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x=plot_df.columns[0], y=plot_df.columns[1], hue='Target', palette='viridis', ax=ax)
    ax.set_title("Before LDA (PCA reduced if >2 features)")
    return fig
