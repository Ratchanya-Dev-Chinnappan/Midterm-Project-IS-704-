# IS704 Classification Analysis - Full Code with EDA and Markdown Output

# --- IMPORTS ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score
)

# --- LOAD & PREPROCESS DATA ---
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# EDA - Summary Statistics
print("\n# Exploratory Data Analysis (EDA)\n")
print(X.describe())

# EDA - Plot Example (Histogram)
import warnings
warnings.filterwarnings('ignore')

sns.histplot(data=X, x='mean radius', hue=y, kde=True)
plt.title('Histogram of Mean Radius by Tumor Type')
plt.xlabel('Mean Radius')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('histogram_mean_radius.png')
plt.clf()

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- MODEL INITIALIZATION & TRAINING ---
logreg = LogisticRegression(max_iter=10000, random_state=42)
lda = LinearDiscriminantAnalysis()

logreg.fit(X_train, y_train)
lda.fit(X_train, y_train)

# --- PREDICTIONS & EVALUATION ---
logreg_preds = logreg.predict(X_test)
lda_preds = lda.predict(X_test)

logreg_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])
lda_auc = roc_auc_score(y_test, lda.predict_proba(X_test)[:, 1])
logreg_cm = confusion_matrix(y_test, logreg_preds)
lda_cm = confusion_matrix(y_test, lda_preds)

results = {
    "Logistic Regression": {
        "Accuracy": np.mean(cross_val_score(logreg, X_scaled, y, cv=5)),
        "Precision": precision_score(y_test, logreg_preds),
        "Recall": recall_score(y_test, logreg_preds),
        "F1-Score": f1_score(y_test, logreg_preds),
        "AUC": logreg_auc,
        "Confusion Matrix": logreg_cm.tolist()
    },
    "LDA": {
        "Accuracy": np.mean(cross_val_score(lda, X_scaled, y, cv=5)),
        "Precision": precision_score(y_test, lda_preds),
        "Recall": recall_score(y_test, lda_preds),
        "F1-Score": f1_score(y_test, lda_preds),
        "AUC": lda_auc,
        "Confusion Matrix": lda_cm.tolist()
    }
}

# --- MARKDOWN OUTPUT ---
print("\n# Classification Report Summary")
for model, metrics in results.items():
    print(f"\n## {model}")
    for metric, value in metrics.items():
        if metric == "Confusion Matrix":
            print(f"- {metric}: {value}")
        else:
            print(f"- {metric}: {value:.4f}")
            
#  Classification Report Summary 

##  Logistic Regression
- Accuracy: 0.9544  
- Precision: 0.9394  
- Recall: 0.9254  
- F1-Score: 0.9323  
- AUC: 0.9813  
- Confusion Matrix: [[43, 4], [5, 62]]

##  LDA
- Accuracy: 0.9545  
- Precision: 0.9275  
- Recall: 0.9552  
- F1-Score: 0.9412  
- AUC: 0.9809  
- Confusion Matrix: [[42, 5], [3, 64]]


        
