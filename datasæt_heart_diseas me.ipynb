import pandas as pd
import numpy as np

# Indlæs datasættet
url_heart_disease = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
col_names_heart = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                   "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

heart_df = pd.read_csv(url_heart_disease, names=col_names_heart, na_values="?")

# Fjern rækker med manglende værdier
heart_df.dropna(inplace=True)

# Tag et sample på fx 50 rækker
heart_df0 = heart_df.sample(n=50, random_state=42)

# Drop nogle kolonner (fx de mere komplekse)
heart_df1 = heart_df0.drop(columns=["ca", "thal", "slope"])

# Kategoriske variable som skal konverteres
binary_mappings = {
    "sex": {1: 1, 0: 0},
    "fbs": {1: 1, 0: 0},
    "exang": {1: 1, 0: 0},
    "restecg": {0: 0, 1: 1, 2: 2}
}

for col, mapping in binary_mappings.items():
    heart_df1[col] = heart_df1[col].replace(mapping)

# Binning af target til 4 klasser 
heart_df1["target"] = pd.cut(heart_df1["target"],
                             bins=[-1, 0, 1, 2, 4],
                             labels=[0, 1, 2, 3],
                             include_lowest=True).astype(int)

# Del dataene op i X og Y
X_heart = heart_df1.drop(columns=["target"])
Y_heart = heart_df1["target"]

# Konverter til NumPy arrays
X_heart_np = X_heart.to_numpy()
Y_heart_np = Y_heart.to_numpy()

# Klasselabels 
class_labels, Y_heart_np = np.unique(heart_df1['target'], return_inverse=True)

# Tjek data
print(heart_df1.head())
