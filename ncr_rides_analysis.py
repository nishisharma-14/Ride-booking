
# 1. IMPORT LIBRARIES
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report



# 2. LOAD DATASET

df = pd.read_csv('ncr_ride_bookings.csv')


# 3. BASIC INFORMATION

print("DATASET SHAPE")
print("Shape:", df.shape)

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATA INFO")
print(df.info())

print("\nSUMMARY STATISTICS")
print(df.describe())

print("\nDUPLICATE ROWS")
print("Duplicate Rows:", df.duplicated().sum())

# Replace 'null' string with real NaN
df = df.replace('null', np.nan)

print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())


# 4. DATA CLEANING

df = df.drop_duplicates()

ride_distance_original = pd.to_numeric(df['Ride Distance'], errors='coerce')

cols_to_drop = [
    'Cancelled Rides by Customer',
    'Reason for cancelling by Customer',
    'Cancelled Rides by Driver',
    'Driver Cancellation Reason',
    'Incomplete Rides',
    'Incomplete Rides Reason'
]
df = df.drop(columns=cols_to_drop)

num_cols = ['Booking Value', 'Ride Distance', 'Avg VTAT', 'Avg CTAT', 'Driver Ratings', 'Customer Rating']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

df['Payment Method'] = df['Payment Method'].fillna(df['Payment Method'].mode()[0])

df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour
df['Hour'] = df['Hour'].fillna(df['Hour'].median())
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['DayOfWeek'] = df['Date'].dt.dayofweek

print("\n--- MISSING VALUES AFTER CLEANING ---")
print(df.isnull().sum())
print("Cleaned Dataset Shape:", df.shape)

# ============================================================
# PROJECT OBJECTIVES AND WHERE EACH IS ANSWERED
# Objective 1: Describe the booking-status distribution.
#              Starts in Section 5, Plot 1.
# Objective 2: Identify the most frequently booked vehicle type.
#              Starts in Section 5, Plot 2.
# Objective 3: Identify peak booking hours.
#              Starts in Section 5, Plot 3.
# Objective 4: Explore ride distance, booking value, and numeric
#              relationships. Starts in Section 5, Plots 4-7.
# Objective 5: Predict completed vs. not completed bookings.
#              Starts in Section 6: Classification Models.
# ============================================================

# ============================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Objective 1, Plot 1: Booking Status Distribution
plt.figure()
df['Booking Status'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Booking Status Percentage')
plt.ylabel('')
plt.show()

# Objective 2, Plot 2: Vehicle Type Count
plt.figure()
sns.countplot(data=df, x='Vehicle Type')
plt.title('Vehicle Type Count')
plt.xticks(rotation=20)
plt.show()

# Objective 3, Plot 3: Peak Booking Hours (Hour of Day)
plt.figure()
sns.countplot(data=df, x='Hour', color='mediumpurple')
plt.title('Bookings by Hour of Day (Peak Hours)')
plt.xlabel('Hour of Day (0 to 23)')
plt.ylabel('Number of Bookings')
plt.show()

# Objective 4, Plot 4: Ride Distance Distribution (original non-missing values)
plt.figure()
sns.histplot(x=ride_distance_original.dropna(), bins=30, color='coral')
plt.title('Ride Distance Distribution')
plt.show()

# Objective 4, Plot 5: Booking Value vs Booking Status Boxplot
plt.figure()
sns.boxplot(data=df, x='Booking Status', y='Booking Value')
plt.title('Booking Value vs Booking Status')
plt.xticks(rotation=15)
plt.show()

# Objective 4, Plot 6: Ride Distance vs Booking Value Scatterplot
plt.figure()
sns.scatterplot(data=df, x='Ride Distance', y='Booking Value', alpha=0.3, color='teal')
plt.title('Ride Distance vs Booking Value')
plt.show()

# Objective 4, Plot 7: Correlations between numeric variables
numeric_data = df.select_dtypes(include='number')
correlation = numeric_data.corr()

plt.figure(figsize=(11, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title('Correlation Heatmap of Numeric Variables')
plt.tight_layout()
plt.show()


# ============================================================
# Objective- 5 (6. CLASSIFICATION: 4 MACHINE LEARNING MODELS)
# ============================================================

X = df.drop(['Booking ID', 'Customer ID', 'Date', 'Time', 'Pickup Location',
             'Drop Location', 'Booking Status'], axis=1)

y = df['Booking Status'].map({
    'Completed': 1,
    'Cancelled by Driver': 0,
    'Cancelled by Customer': 0,
    'No Driver Found': 0,
    'Incomplete': 0
})

X = pd.get_dummies(X, drop_first=True, dtype=int)
# Remove rows whose booking status was not included in the target mapping.
valid_target = y.notna()
X = X.loc[valid_target]
y = y.loc[valid_target].astype(int)

print(X.shape)
print(y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape)
print(X_test.shape)

# Logistic regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\nDecision Tree:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))
print("F1 Score:", f1_score(y_test, y_pred_dt))

# Random forest
rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

print("\nKNN:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Precision:", precision_score(y_test, y_pred_knn))
print("Recall:", recall_score(y_test, y_pred_knn))
print("F1 Score:", f1_score(y_test, y_pred_knn))


# ============================================================
# 7. ACCURACY COMPARISON & BEST MODEL EVALUATION
# ============================================================
# Compare Accuracies with a simple Bar Plot
models = ['Logistic Reg', 'Decision Tree', 'Random Forest', 'KNN']
accuracies = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_dt),
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_knn)
]

plt.figure()
plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'])
plt.title('Classification Models Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()

# Best Model Evaluation: Random Forest has the strongest scores from your results
print("\n==============================================")
print("BEST-FIT MODEL EVALUATION: RANDOM FOREST")
print("==============================================")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1 Score :", f1_score(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))


# Plot Confusion Matrix

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',)

plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Booking Status')
plt.ylabel('Actual Booking Status')
plt.show()


# FINAL CONCLUSION

# The analysis covers booking status, vehicle popularity, peak hours,
# ride-distance/fare patterns, and correlations between numeric variables.
# In the scores shared for the current run, Random Forest was the best overall:
# Accuracy = 98.40%, Precision = 97.85%, Recall = 99.61%, F1 = 98.72%.
# Recheck these values against the latest run before including them in a report.

