from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

def apply_lda(X_train, y_train, X_test, n_components=2):
    """
    Applies Linear Discriminant Analysis to the given datasets.
    """
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    
    # Create DataFrames for easier plotting
    cols = [f'LDA_Component_{i+1}' for i in range(n_components)]
    X_train_lda = pd.DataFrame(X_train_lda, columns=cols)
    X_test_lda = pd.DataFrame(X_test_lda, columns=cols)
    
    return lda, X_train_lda, X_test_lda
