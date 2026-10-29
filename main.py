# 1. IMPORT REQUIRED LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# 2. LOAD DATASET

df = pd.read_csv("C:/Users/Asus/Downloads/9ef84268-d588-465a-a308-a864a43d0070 (3).csv")

print(df.head())
print(df.tail())
df.info()
print(df.describe(include="all"))

# 3. COLUMN NAME CLEANING AND TYPE CONVERSION

df.columns = df.columns.str.strip()

df.rename(columns={
    "Min_x0020_Price": "Min_Price",
    "Max_x0020_Price": "Max_Price",
    "Modal_x0020_Price": "Modal_Price"
}, inplace=True)

categorical_columns = [
    "State", "District", "Market",
    "Commodity", "Variety", "Grade"
]

numeric_columns = [
    "Min_Price", "Max_Price", "Modal_Price"
]

for col in categorical_columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Arrival_Date"] = pd.to_datetime(
    df["Arrival_Date"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

print("\nData Types:")
print(df.dtypes)

# 4. DUPLICATE VALUE DETECTION AND REMOVAL

print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# 5. MISSING VALUE ANALYSIS AND HANDLING

print("\nMissing Values Before Handling:")
print(df.isnull().sum())

df.dropna(subset=["Modal_Price"], inplace=True)

for col in categorical_columns:
    df[col] = df[col].fillna("Unknown")

for col in ["Min_Price", "Max_Price"]:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing Values After Handling:")
print(df.isnull().sum())

print("\nFinal Dataset Shape:", df.shape)

# 6. CORRELATION AND COVARIANCE ANALYSIS

print("\nCorrelation Matrix:")
print(df[numeric_columns].corr())

print("\nCovariance Matrix:")
print(df[numeric_columns].cov())

# 7. SKEWNESS AND KURTOSIS ANALYSIS

print("\nModal Price Skewness:", df["Modal_Price"].skew())
print("Modal Price Kurtosis:", df["Modal_Price"].kurt())

# 8. OUTLIER DETECTION USING IQR

Q1 = df["Modal_Price"].quantile(0.25)
Q3 = df["Modal_Price"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["Modal_Price"] < lower_bound) |
    (df["Modal_Price"] > upper_bound)
]

print("\nModal Price Outliers:", len(outliers))

# 9. HISTOGRAM OF MODAL PRICE

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Modal_Price"],
    bins=25,
    kde=True
)

plt.title("Distribution of Modal Prices")
plt.xlabel("Modal Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 10. BOXPLOT OF MODAL PRICE

plt.figure(figsize=(8, 4))

sns.boxplot(x=df["Modal_Price"])

plt.title("Boxplot of Modal Prices")
plt.xlabel("Modal Price")
plt.tight_layout()
plt.show()

# 11. CORRELATION HEATMAP

plt.figure(figsize=(7, 5))

sns.heatmap(
    df[numeric_columns].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 12. TOP 10 HIGHEST AND LOWEST PRICED COMMODITIES

commodity_prices = df.groupby("Commodity")["Modal_Price"].mean()

top_10 = commodity_prices.nlargest(10)
bottom_10 = commodity_prices.nsmallest(10)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

top_10.sort_values().plot(kind="barh", ax=axes[0])
axes[0].set_title("Top 10 Highest-Priced Commodities")
axes[0].set_xlabel("Average Modal Price")
axes[0].set_ylabel("Commodity")

bottom_10.sort_values(ascending=False).plot(kind="barh", ax=axes[1])
axes[1].set_title("Top 10 Lowest-Priced Commodities")
axes[1].set_xlabel("Average Modal Price")
axes[1].set_ylabel("Commodity")

plt.tight_layout()
plt.show()

# 13. TOP 10 STATES BY AVERAGE MODAL PRICE

top_states = (
    df.groupby("State")["Modal_Price"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 States:")
print(top_states)

top_states.sort_values().plot(
    kind="barh", figsize=(8, 5)
)

plt.title("Top 10 States by Average Agricultural Market Price")
plt.xlabel("Average Modal Price")
plt.tight_layout()
plt.show()

# 14. CREATE CLASSIFICATION TARGET

df["Price_Category"] = pd.qcut(
    df["Modal_Price"],
    q=3,
    labels=["Low", "Medium", "High"],
    duplicates="drop"
)

print("\nPrice Category Distribution:")
print(df["Price_Category"].value_counts())

# 15. PREPARE CLASSIFICATION DATA

class_features = [
    "State", "District", "Market",
    "Commodity", "Variety", "Grade",
    "Min_Price", "Max_Price"
]

X = df[class_features].copy()
Y = df["Price_Category"]

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    dtype=int
)

X = X.astype(float)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.30,
    random_state=42,
    stratify=Y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 16. CLASSIFICATION MODEL 1: KNN

k = max(1, round(np.sqrt(len(X_train))))

model1 = KNeighborsClassifier(n_neighbors=k)
model1.fit(X_train, Y_train)

y_pred1 = model1.predict(X_test)

print("\nKNN CLASSIFICATION")
print("Accuracy:", accuracy_score(Y_test, y_pred1))
print("Confusion Matrix:")
print(confusion_matrix(
    Y_test, y_pred1, labels=["Low", "Medium", "High"]
))
print("Classification Report:")
print(classification_report(Y_test, y_pred1, zero_division=0))

# 17. CLASSIFICATION MODEL 2: NAIVE BAYES

model2 = GaussianNB()
model2.fit(X_train, Y_train)

y_pred2 = model2.predict(X_test)

print("\nNAIVE BAYES CLASSIFICATION")
print("Accuracy:", accuracy_score(Y_test, y_pred2))
print("Confusion Matrix:")
print(confusion_matrix(
    Y_test, y_pred2, labels=["Low", "Medium", "High"]
))
print("Classification Report:")
print(classification_report(Y_test, y_pred2, zero_division=0))

# 18. CLASSIFICATION MODEL 3: LOGISTIC REGRESSION

model3 = LogisticRegression(max_iter=2000)
model3.fit(X_train, Y_train)

y_pred3 = model3.predict(X_test)

print("\nLOGISTIC REGRESSION CLASSIFICATION")
print("Accuracy:", accuracy_score(Y_test, y_pred3))
print("Confusion Matrix:")
print(confusion_matrix(
    Y_test, y_pred3, labels=["Low", "Medium", "High"]
))
print("Classification Report:")
print(classification_report(Y_test, y_pred3, zero_division=0))

# 19. CLASSIFICATION MODEL COMPARISON

classification_models = [
    "KNN", "Naive Bayes", "Logistic Regression"
]

classification_predictions = [
    y_pred1, y_pred2, y_pred3
]

class_results = []

for name, pred in zip(
    classification_models, classification_predictions
):
    class_results.append({
        "Model": name,
        "Accuracy": accuracy_score(Y_test, pred),
        "Precision": precision_score(
            Y_test, pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            Y_test, pred, average="weighted", zero_division=0
        ),
        "F1 Score": f1_score(
            Y_test, pred, average="weighted", zero_division=0
        )
    })

class_results_df = pd.DataFrame(class_results)

print("\nCLASSIFICATION COMPARISON")
print(class_results_df.to_string(index=False))

class_results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(kind="bar", figsize=(10, 6))

plt.title("Classification Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 20. SIMPLE LINEAR REGRESSION

# One independent variable: Min_Price

X_simple = df[["Min_Price"]]
Y_reg = df["Modal_Price"]

X_train_s, X_test_s, Y_train_s, Y_test_s = train_test_split(
    X_simple, Y_reg,
    test_size=0.30,
    random_state=42
)

simple_model = LinearRegression()

simple_model.fit(X_train_s, Y_train_s)

pred_simple = simple_model.predict(X_test_s)

print("\nSIMPLE LINEAR REGRESSION")
print("MAE:", mean_absolute_error(Y_test_s, pred_simple))
print("MSE:", mean_squared_error(Y_test_s, pred_simple))
print("RMSE:", np.sqrt(mean_squared_error(Y_test_s, pred_simple)))
print("R2 Score:", r2_score(Y_test_s, pred_simple))

# Plot actual data and regression line

sort_idx = np.argsort(X_test_s["Min_Price"].to_numpy())

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test_s["Min_Price"],
    Y_test_s,
    alpha=0.5,
    label="Actual Prices"
)

plt.plot(
    X_test_s["Min_Price"].to_numpy()[sort_idx],
    pred_simple[sort_idx],
    label="Regression Line"
)

plt.title("Minimum Price vs Modal Price: Regression Analysis")
plt.xlabel("Minimum Price")
plt.ylabel("Modal Price")
plt.legend()
plt.tight_layout()
plt.show()

# 21. MULTIPLE LINEAR REGRESSION

# Two independent variables: Min_Price and Max_Price

X_multiple = df[["Min_Price", "Max_Price"]]

X_train_m, X_test_m, Y_train_m, Y_test_m = train_test_split(
    X_multiple, Y_reg,
    test_size=0.30,
    random_state=42
)

multiple_model = LinearRegression()

multiple_model.fit(X_train_m, Y_train_m)

pred_multiple = multiple_model.predict(X_test_m)

print("\nMULTIPLE LINEAR REGRESSION")
print("MAE:", mean_absolute_error(Y_test_m, pred_multiple))
print("MSE:", mean_squared_error(Y_test_m, pred_multiple))
print("RMSE:", np.sqrt(mean_squared_error(Y_test_m, pred_multiple)))
print("R2 Score:", r2_score(Y_test_m, pred_multiple))

# REGRESSION MODEL COMPARISON

print("\nSimple Linear Regression")
print("MAE:", mean_absolute_error(Y_test_s, pred_simple))
print("MSE:", mean_squared_error(Y_test_s, pred_simple))
print("RMSE:", np.sqrt(mean_squared_error(Y_test_s, pred_simple)))
print("R2 Score:", r2_score(Y_test_s, pred_simple))

print("\nMultiple Linear Regression")
print("MAE:", mean_absolute_error(Y_test_m, pred_multiple))
print("MSE:", mean_squared_error(Y_test_m, pred_multiple))
print("RMSE:", np.sqrt(mean_squared_error(Y_test_m, pred_multiple)))
print("R2 Score:", r2_score(Y_test_m, pred_multiple))

 
