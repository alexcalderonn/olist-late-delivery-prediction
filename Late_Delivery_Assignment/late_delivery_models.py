
import pandas as pd

# Load the Olist dataset
df = pd.read_csv("olist_late_delivery.csv")

# Display the first five orders
print(df.head())

# Check the number of rows and columns
print("Dataset shape:", df.shape)

# Check the column names
print(df.columns)

# Convert purchase timestamp to datetime
df["order_purchase_timestamp"] = pd.to_datetime(
    df["order_purchase_timestamp"]
)

# Sort orders by purchase date
df = df.sort_values("order_purchase_timestamp")

# Check the updated dataset
print("Updated dataset shape:", df.shape)
print("Earliest order:", df["order_purchase_timestamp"].min())
print("Latest order:", df["order_purchase_timestamp"].max())


# Split the data by purchase date
# 60% training, 20% validation, 20% testing

train_end = int(len(df) * 0.60)
val_end = int(len(df) * 0.80)

train = df.iloc[:train_end].copy()
validation = df.iloc[train_end:val_end].copy()
test = df.iloc[val_end:].copy()

# Check the number of orders in each group
print("Training orders:", len(train))
print("Validation orders:", len(validation))
print("Testing orders:", len(test))

# Check late delivery rates
print("Training late rate:", train["late_delivery"].mean())
print("Validation late rate:", validation["late_delivery"].mean())
print("Testing late rate:", test["late_delivery"].mean())


# Step 5: Prepare variables for modeling

# Select the predictor columns
features = [
    "purchase_month",
    "purchase_day",
    "purchase_hour",
    "estimated_days",
    "customer_state"
]

# Separate predictors and target
X_train = train[features]
y_train = train["late_delivery"]

X_val = validation[features]
y_val = validation["late_delivery"]

X_test = test[features]
y_test = test["late_delivery"]

# Convert customer state into numeric columns
X_train = pd.get_dummies(X_train)
X_val = pd.get_dummies(X_val)
X_test = pd.get_dummies(X_test)

# Make sure all datasets have the same columns
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Check the results
print("Training predictors:", X_train.shape)
print("Validation predictors:", X_val.shape)
print("Testing predictors:", X_test.shape)
print("Training target:", y_train.shape)


# Step 6: Logistic Regression Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Create the model
log_model = LogisticRegression(max_iter=1000)

# Train the model using training data
log_model.fit(X_train, y_train)

# Predict late deliveries on validation data
log_predictions = log_model.predict(X_val)

# Evaluate the model
print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_val, log_predictions))
print("Precision:", precision_score(y_val, log_predictions, zero_division=0))
print("Recall:", recall_score(y_val, log_predictions, zero_division=0))
print("F1 Score:", f1_score(y_val, log_predictions, zero_division=0))


# Step 7: Decision Tree Model

from sklearn.tree import DecisionTreeClassifier

# Create the Decision Tree model
tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Train the model
tree_model.fit(X_train, y_train)

# Predict late deliveries on validation data
tree_predictions = tree_model.predict(X_val)

# Evaluate the model
print("\nDecision Tree Results")
print("Accuracy:", accuracy_score(y_val, tree_predictions))
print("Precision:", precision_score(y_val, tree_predictions, zero_division=0))
print("Recall:", recall_score(y_val, tree_predictions, zero_division=0))
print("F1 Score:", f1_score(y_val, tree_predictions, zero_division=0))


# Step 8: Try models with balanced class weights

# Balanced Logistic Regression
log_balanced = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

log_balanced.fit(X_train, y_train)
log_balanced_predictions = log_balanced.predict(X_val)

print("\nBalanced Logistic Regression")
print("Accuracy:", accuracy_score(y_val, log_balanced_predictions))
print("Precision:", precision_score(y_val, log_balanced_predictions, zero_division=0))
print("Recall:", recall_score(y_val, log_balanced_predictions, zero_division=0))
print("F1 Score:", f1_score(y_val, log_balanced_predictions, zero_division=0))


# Balanced Decision Tree
tree_balanced = DecisionTreeClassifier(
    max_depth=5,
    class_weight="balanced",
    random_state=42
)

tree_balanced.fit(X_train, y_train)
tree_balanced_predictions = tree_balanced.predict(X_val)

print("\nBalanced Decision Tree")
print("Accuracy:", accuracy_score(y_val, tree_balanced_predictions))
print("Precision:", precision_score(y_val, tree_balanced_predictions, zero_division=0))
print("Recall:", recall_score(y_val, tree_balanced_predictions, zero_division=0))
print("F1 Score:", f1_score(y_val, tree_balanced_predictions, zero_division=0))


# Step 9: Final model evaluation on test data

# Predict late deliveries on the test set
test_predictions = log_balanced.predict(X_test)

print("\nFinal Test Results - Balanced Logistic Regression")
print("Accuracy:", accuracy_score(y_test, test_predictions))
print("Precision:", precision_score(y_test, test_predictions, zero_division=0))
print("Recall:", recall_score(y_test, test_predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, test_predictions, zero_division=0))

# Display confusion matrix
from sklearn.metrics import confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))


# Step 10: Compare ROC-AUC and PR-AUC

from sklearn.metrics import roc_auc_score, average_precision_score

# Get prediction probabilities from both models
log_prob = log_balanced.predict_proba(X_val)[:, 1]
tree_prob = tree_balanced.predict_proba(X_val)[:, 1]

# Calculate ROC-AUC
log_roc = roc_auc_score(y_val, log_prob)
tree_roc = roc_auc_score(y_val, tree_prob)

# Calculate PR-AUC (Average Precision)
log_pr = average_precision_score(y_val, log_prob)
tree_pr = average_precision_score(y_val, tree_prob)

# Display results
print("\nBalanced Logistic Regression")
print("ROC-AUC:", log_roc)
print("PR-AUC:", log_pr)

print("\nBalanced Decision Tree")
print("ROC-AUC:", tree_roc)
print("PR-AUC:", tree_pr)


# Step 11: Compare model performance

import matplotlib.pyplot as plt

models = ["Logistic Regression", "Decision Tree"]

roc_scores = [log_roc, tree_roc]
pr_scores = [log_pr, tree_pr]

# Create comparison chart
plt.figure(figsize=(8, 5))

plt.plot(models, roc_scores, marker="o", label="ROC-AUC")
plt.plot(models, pr_scores, marker="o", label="PR-AUC")

plt.title("Late Delivery Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)

plt.savefig("model_comparison.png")
plt.close()

# Step 12: Final Test Comparison

from sklearn.metrics import confusion_matrix, roc_auc_score

# Balanced Logistic Regression predictions
log_test_predictions = log_balanced.predict(X_test)
log_test_prob = log_balanced.predict_proba(X_test)[:, 1]

# Balanced Decision Tree predictions
tree_test_predictions = tree_balanced.predict(X_test)
tree_test_prob = tree_balanced.predict_proba(X_test)[:, 1]

# Logistic Regression test results
print("\nFINAL TEST: LOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, log_test_predictions))
print("Precision:", precision_score(y_test, log_test_predictions, zero_division=0))
print("Recall:", recall_score(y_test, log_test_predictions, zero_division=0))
print("F1:", f1_score(y_test, log_test_predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, log_test_prob))
print("Confusion Matrix:")
print(confusion_matrix(y_test, log_test_predictions))

# Decision Tree test results
print("\nFINAL TEST: DECISION TREE")
print("Accuracy:", accuracy_score(y_test, tree_test_predictions))
print("Precision:", precision_score(y_test, tree_test_predictions, zero_division=0))
print("Recall:", recall_score(y_test, tree_test_predictions, zero_division=0))
print("F1:", f1_score(y_test, tree_test_predictions, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, tree_test_prob))
print("Confusion Matrix:")
print(confusion_matrix(y_test, tree_test_predictions))
