import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# 🔥 Load your balanced dataset
df = pd.read_csv("tna_scores_dataset1.csv")  # your balanced dataset with FDP_need

# ✅ Ensure FDP_need is treated as integer
df["FDP_need"] = df["FDP_need"].astype(int)

# ✅ Split into features and target
X = df.drop("FDP_need", axis=1)
y = df["FDP_need"]

# ✅ Train-test split with stratification to preserve balance
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)

# ✅ Train RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ✅ Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# ✅ Save model
with open("tna_model1.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Model saved as tna_model1.pkl")
