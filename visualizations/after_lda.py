import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_lda_data(X_lda, y):
    """
    Plots the LDA transformed data.
    """
    plot_df = X_lda.copy()
    if isinstance(y, pd.Series):
        plot_df['Target'] = y.values
    else:
        plot_df['Target'] = y
        
    fig, ax = plt.subplots(figsize=(8, 6))
    if X_lda.shape[1] == 1:
        # 1D LDA scatter plot against zeros
        sns.scatterplot(x=plot_df.iloc[:, 0], y=[0]*len(plot_df), hue=plot_df['Target'], palette='viridis', ax=ax)
    else:
        sns.scatterplot(data=plot_df, x=plot_df.columns[0], y=plot_df.columns[1], hue='Target', palette='viridis', ax=ax)
    
    ax.set_title("After LDA")
    return fig
