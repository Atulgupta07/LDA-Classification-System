# LDA Classification System

An interactive Streamlit web application designed to teach and demonstrate how **Linear Discriminant Analysis (LDA)** improves class separability and downstream classification performance using the Iris dataset as a running example.

## Pipeline Overview

1. **Raw Dataset:** Explore the Iris dataset characteristics and summary statistics.
2. **Preprocessing:** Handle missing values, encode targets, perform stratified train/test splits, and apply Standard Scaling to prevent data leakage.
3. **LDA & Visualizations:** Apply LDA to the scaled dataset and visually compare the feature space (using PCA for raw data if needed) before and after the LDA transformation.
4. **Classification Models:** Train multiple classifiers (Logistic Regression, Support Vector Machines, Random Forest) on both the raw scaled data and the LDA-transformed data.
5. **Evaluation:** Compare the performance of the models using metrics like Accuracy, Precision, Recall, F1-Score, and interactive Confusion Matrices.
6. **Prediction:** Input custom feature values to see real-time predictions from all trained models simultaneously.

## Project Structure

```text
LDA-Classification-System/
├── app.py                     # Main Streamlit application
├── data/
│   └── iris.csv               # Dataset (auto-generated on first run)
├── src/
│   ├── preprocessing.py       # Data loading, summary, and scaling (Phase 1)
│   ├── lda.py                 # LDA transformation logic (Phase 2)
│   ├── classification.py      # Classifier training (Phase 4)
│   ├── evaluation.py          # Metrics and evaluation logic (Phase 5)
│   └── prediction.py          # Real-time inference (Phase 6)
├── visualizations/
│   ├── before_lda.py          # Pre-LDA data visualization (Phase 3)
│   ├── after_lda.py           # Post-LDA data visualization (Phase 3)
│   └── evaluation.py          # Metric plots and confusion matrices (Phase 5)
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Setup & Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd LDA-Classification-System
   ```

2. **Install the required dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Usage

Once the app is running, use the tabs at the top to navigate through the data pipeline sequentially:
- Start at the **Dataset** tab to view the raw data.
- Move to **Preprocessing** to define the test split size and scale the data.
- Proceed to **LDA & Visualizations** to apply the transformation and view the 2D scatter plots.
- Head to **Classification & Evaluation** to train the models and compare their metrics.
- Finally, use the **Prediction** tab to input your own values and test the trained models.
