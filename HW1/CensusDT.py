import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Setting file paths.
train_path = "./CensusData/census-income.data"
test_path = "./CensusData/census-income.test"

# Loading data.
def load_data(path):
    df = pd.read_csv(
        path,
        names=[f"col{i}" for i in range(42)],
        header=None,
        skipinitialspace=True,
        na_values=["?"],
        engine="python"
    )
    df["col41"] = df["col41"].str.replace(".", "", regex=False).str.strip()
    return df

train = load_data(train_path)
test = load_data(test_path)

# Preprocessing the data.
categorical_cols = train.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

X_train = train.drop(columns=["col41"])
y_train = train["col41"]
X_test = test.drop(columns=["col41"])
y_test = test["col41"]

# Training a decision tree with varying depths.
train_accuracies = []
test_accuracies = []

for depth in range(2, 11):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    
    train_accuracies.append((depth, train_acc))
    test_accuracies.append((depth, test_acc))
    print(f"Depth {depth}: Training Accuracy = {train_acc:.4f}, Testing Accuracy = {test_acc:.4f}")

# Selecting optimal depth based on the highest testing accuracy.
optimal_depth = max(test_accuracies, key=lambda x: x[1])[0]
print(f"Optimal Depth: {optimal_depth}")

# Training the final model.
final_clf = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
final_clf.fit(X_train, y_train)

# Outputting the final model's training and testing accuracy.
train_acc_opt = accuracy_score(y_train, final_clf.predict(X_train))
print(f"Final Model Training Accuracy: {train_acc_opt:.4f}")
final_test_acc = accuracy_score(y_test, final_clf.predict(X_test))
print(f"Final Model Testing Accuracy: {final_test_acc:.4f}")

# Checking for overfitting.
diff = train_acc_opt - final_test_acc
print(f"Overfitting Gap (Training - Testing Accuracy): {diff:.4f}")
