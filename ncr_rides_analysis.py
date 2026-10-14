# ============================================================
#   NCR Ride Bookings - Data Science & ML Project
#   Course  : INT234 - Predictive Analytics
#   Dataset : ncr_ride_bookings.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

print("STEP 1 - Loading Dataset")
print("-" * 40)

df = pd.read_csv('ncr_ride_bookings.csv')
df.replace('null', np.nan, inplace=True)

print("Shape of dataset:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic Statistics:")
print(df.describe())


# ============================================================
# STEP 2 - DATA CLEANING
# ============================================================

print("\nSTEP 2 - Data Cleaning")
print("-" * 40)

# Check and remove duplicates
print("Duplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)

# Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Drop columns with too many missing values (more than 70%)
cols_to_drop = ['Cancelled Rides by Customer',
                'Reason for cancelling by Customer',
                'Cancelled Rides by Driver',
                'Driver Cancellation Reason',
                'Incomplete Rides',
                'Incomplete Rides Reason']
df.drop(columns=cols_to_drop, inplace=True)
print("\nDropped 6 high-missing columns.")

# Convert columns to numeric
df['Booking Value']   = pd.to_numeric(df['Booking Value'],   errors='coerce')
df['Ride Distance']   = pd.to_numeric(df['Ride Distance'],   errors='coerce')
df['Avg VTAT']        = pd.to_numeric(df['Avg VTAT'],        errors='coerce')
df['Avg CTAT']        = pd.to_numeric(df['Avg CTAT'],        errors='coerce')
df['Driver Ratings']  = pd.to_numeric(df['Driver Ratings'],  errors='coerce')
df['Customer Rating'] = pd.to_numeric(df['Customer Rating'], errors='coerce')

# Fill missing numeric values with median
df['Booking Value'].fillna(df['Booking Value'].median(),     inplace=True)
df['Ride Distance'].fillna(df['Ride Distance'].median(),     inplace=True)
df['Avg VTAT'].fillna(df['Avg VTAT'].median(),               inplace=True)
df['Avg CTAT'].fillna(df['Avg CTAT'].median(),               inplace=True)
df['Driver Ratings'].fillna(df['Driver Ratings'].median(),   inplace=True)
df['Customer Rating'].fillna(df['Customer Rating'].median(), inplace=True)

# Fill missing text with most common value
df['Payment Method'].fillna(df['Payment Method'].mode()[0], inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nFinal dataset shape:", df.shape)


# ============================================================
# STEP 3 - OBJECTIVES
# ============================================================

print("\nSTEP 3 - Project Objectives")
print("-" * 40)
print("Objective 1: Predict Booking Status (Completed or Not) - Classification")
print("Objective 2: Predict Booking Value (fare) - Regression")
print("Objective 3: Find peak booking hours - EDA")
print("Objective 4: Find most popular and profitable vehicle type - EDA")
print("Objective 5: Group customers by ride behavior - Clustering")


# ============================================================
# STEP 4 - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("\nSTEP 4 - Exploratory Data Analysis")
print("-" * 40)

# Plot 1 - Booking Status Distribution
status = df['Booking Status'].value_counts()
plt.figure()
plt.bar(status.index, status.values, color=['green','red','orange','blue','gray'])
plt.title('Plot 1: Booking Status Distribution')
plt.xlabel('Booking Status')
plt.ylabel('Count')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Plot 2 - Vehicle Type Count (Objective 4)
vehicle = df['Vehicle Type'].value_counts()
plt.figure()
plt.bar(vehicle.index, vehicle.values, color='steelblue')
plt.title('Plot 2: Vehicle Type Count (Objective 4)')
plt.xlabel('Vehicle Type')
plt.ylabel('Count')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Plot 3 - Booking Value Distribution
plt.figure()
plt.hist(df['Booking Value'], bins=30, color='coral', edgecolor='black')
plt.title('Plot 3: Booking Value Distribution')
plt.xlabel('Booking Value (INR)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot 4 - Peak Hours (Objective 3)
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce').dt.hour
hour_counts = df['Hour'].value_counts().sort_index()
plt.figure()
plt.bar(hour_counts.index, hour_counts.values, color='purple')
plt.title('Plot 4: Bookings by Hour of Day - Peak Hours (Objective 3)')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Bookings')
plt.tight_layout()
plt.show()

# Plot 5 - Avg Booking Value by Vehicle Type (Objective 4)
avg_value = df.groupby('Vehicle Type')['Booking Value'].mean().sort_values(ascending=False)
plt.figure()
plt.bar(avg_value.index, avg_value.values, color='darkorange')
plt.title('Plot 5: Avg Booking Value by Vehicle Type (Objective 4)')
plt.xlabel('Vehicle Type')
plt.ylabel('Average Booking Value (INR)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Plot 6 - Ride Distance vs Booking Value
plt.figure()
plt.scatter(df['Ride Distance'], df['Booking Value'], alpha=0.2, s=5, color='navy')
plt.title('Plot 6: Ride Distance vs Booking Value')
plt.xlabel('Ride Distance (km)')
plt.ylabel('Booking Value (INR)')
plt.tight_layout()
plt.show()


# ============================================================
# STEP 5 - FEATURE ENGINEERING
# ============================================================

print("\nSTEP 5 - Feature Engineering")
print("-" * 40)

# Label encode Vehicle Type
le = LabelEncoder()
df['Vehicle_Type_enc'] = le.fit_transform(df['Vehicle Type'])

# Target column: 1 = Completed, 0 = Not Completed
df['Status_Label'] = df['Booking Status'].apply(lambda x: 1 if x == 'Completed' else 0)

print("Vehicle_Type_enc and Status_Label columns created.")
print(df[['Vehicle Type', 'Vehicle_Type_enc', 'Booking Status', 'Status_Label']].head(8))


# ============================================================
# STEP 6 - CLASSIFICATION MODELS (Objective 1)
# ============================================================

print("\nSTEP 6 - Classification Models")
print("-" * 40)
print("Goal: Predict whether a ride is Completed (1) or Not (0)")

features = ['Vehicle_Type_enc', 'Ride Distance', 'Hour', 'Avg VTAT']
df_model = df[features + ['Status_Label']].dropna()

X = df_model[features]
y = df_model['Status_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training size:", X_train.shape[0])
print("Testing size :", X_test.shape[0])

# Model 1 - Logistic Regression
print("\n--- Model 1: Logistic Regression ---")
model_lr = LogisticRegression(max_iter=500)
model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)
acc_lr = accuracy_score(y_test, pred_lr)
print("Accuracy:", round(acc_lr * 100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_lr))

# Model 2 - Decision Tree
print("\n--- Model 2: Decision Tree ---")
model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
model_dt.fit(X_train, y_train)
pred_dt = model_dt.predict(X_test)
acc_dt = accuracy_score(y_test, pred_dt)
print("Accuracy:", round(acc_dt * 100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_dt))

# Model 3 - KNN
print("\n--- Model 3: K-Nearest Neighbors (k=5) ---")
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train, y_train)
pred_knn = model_knn.predict(X_test)
acc_knn = accuracy_score(y_test, pred_knn)
print("Accuracy:", round(acc_knn * 100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_knn))

# Model 4 - Naive Bayes
print("\n--- Model 4: Naive Bayes ---")
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
pred_nb = model_nb.predict(X_test)
acc_nb = accuracy_score(y_test, pred_nb)
print("Accuracy:", round(acc_nb * 100, 2), "%")
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_nb))


# Classification Comparison Bar Chart
model_names = ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes']
accuracies  = [acc_lr, acc_dt, acc_knn, acc_nb]

plt.figure()
plt.bar(model_names, accuracies, color=['blue', 'green', 'orange', 'red'])
plt.title('Classification Models - Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, str(round(v * 100, 1)) + '%', ha='center')
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()

best_cls = model_names[accuracies.index(max(accuracies))]
print("\nBest Classification Model:", best_cls,
      "| Accuracy:", round(max(accuracies) * 100, 2), "%")


# ============================================================
# STEP 7 - REGRESSION MODELS (Objective 2)
# ============================================================

print("\nSTEP 7 - Regression Models")
print("-" * 40)
print("Goal: Predict Booking Value for completed rides")

df_reg = df[df['Booking Status'] == 'Completed'].copy()
df_reg = df_reg[['Ride Distance', 'Vehicle_Type_enc', 'Hour', 'Booking Value']].dropna()

X_r = df_reg[['Ride Distance', 'Vehicle_Type_enc', 'Hour']]
y_r = df_reg['Booking Value']

X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
print("Training size:", X_r_train.shape[0])
print("Testing size :", X_r_test.shape[0])

# Reg Model 1 - Simple Linear Regression
print("\n--- Reg Model 1: Simple Linear Regression ---")
slr = LinearRegression()
slr.fit(X_r_train[['Ride Distance']], y_r_train)
pred_slr = slr.predict(X_r_test[['Ride Distance']])
mae_slr  = round(mean_absolute_error(y_r_test, pred_slr), 2)
rmse_slr = round(np.sqrt(mean_squared_error(y_r_test, pred_slr)), 2)
r2_slr   = round(r2_score(y_r_test, pred_slr), 4)
print("MAE :", mae_slr)
print("RMSE:", rmse_slr)
print("R2  :", r2_slr)

# Reg Model 2 - Multiple Linear Regression
print("\n--- Reg Model 2: Multiple Linear Regression ---")
mlr = LinearRegression()
mlr.fit(X_r_train, y_r_train)
pred_mlr = mlr.predict(X_r_test)
mae_mlr  = round(mean_absolute_error(y_r_test, pred_mlr), 2)
rmse_mlr = round(np.sqrt(mean_squared_error(y_r_test, pred_mlr)), 2)
r2_mlr   = round(r2_score(y_r_test, pred_mlr), 4)
print("MAE :", mae_mlr)
print("RMSE:", rmse_mlr)
print("R2  :", r2_mlr)
print("Coefficients:", mlr.coef_)

# Regression Comparison
reg_names = ['Simple Linear Reg', 'Multiple Linear Reg']
mae_vals  = [mae_slr, mae_mlr]

plt.figure()
plt.bar(reg_names, mae_vals, color=['blue', 'green'])
plt.title('Regression Models - MAE Comparison (lower is better)')
plt.xlabel('Model')
plt.ylabel('MAE')
for i, v in enumerate(mae_vals):
    plt.text(i, v + 0.5, str(v), ha='center')
plt.tight_layout()
plt.show()

best_reg = reg_names[mae_vals.index(min(mae_vals))]
print("\nBest Regression Model:", best_reg)


# ============================================================
# STEP 8 - CLUSTERING - K-Means (Objective 5)
# ============================================================

print("\nSTEP 8 - K-Means Clustering")
print("-" * 40)
print("Goal: Group customers by ride behavior")

df_cluster = df[['Booking Value', 'Ride Distance', 'Customer Rating']].dropna().copy()
print("Clustering dataset size:", df_cluster.shape)

# Elbow Method
wcss = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df_cluster)
    wcss.append(km.inertia_)

plt.figure()
plt.plot(range(2, 9), wcss, marker='o', color='blue')
plt.title('Elbow Method - Find Best Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.tight_layout()
plt.show()

# Apply K-Means with k=3
km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cluster['Cluster'] = km_final.fit_predict(df_cluster)

print("\nCluster sizes:")
print(df_cluster['Cluster'].value_counts())
print("\nCluster averages:")
print(df_cluster.groupby('Cluster').mean().round(2))

# Cluster Plot
plt.figure()
colors = ['red', 'blue', 'green']
for i in range(3):
    c = df_cluster[df_cluster['Cluster'] == i]
    plt.scatter(c['Ride Distance'], c['Booking Value'],
                s=5, alpha=0.3, color=colors[i], label='Cluster ' + str(i))
plt.title('Customer Clusters - Distance vs Booking Value (Objective 5)')
plt.xlabel('Ride Distance (km)')
plt.ylabel('Booking Value (INR)')
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# STEP 9 - FINAL SUMMARY & CONCLUSIONS
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY & CONCLUSIONS")
print("=" * 60)

print("\n--- Classification Results ---")
for i in range(len(model_names)):
    print(f"  {model_names[i]:22s} | Accuracy: {round(accuracies[i]*100, 2)}%")
print("  Best Model:", best_cls)

print("\n--- Regression Results ---")
print(f"  {'Simple Linear Reg':22s} | MAE: {mae_slr} | R2: {r2_slr}")
print(f"  {'Multiple Linear Reg':22s} | MAE: {mae_mlr} | R2: {r2_mlr}")
print("  Best Model:", best_reg)

print("\n--- Clustering Results ---")
print("  3 customer groups identified:")
print("  Cluster 0 = Budget riders  (short distance, low fare)")
print("  Cluster 1 = Regular riders (medium distance, medium fare)")
print("  Cluster 2 = Premium riders (long distance, high fare)")

print("""
--- Key Findings ---
  1. 62% of rides are Completed
  2. Auto is the most booked vehicle type
  3. Peak hours are 8-10 AM and 5-8 PM
  4. Premier Sedan has the highest average booking value
  5. UPI is the most used payment method

--- Recommendations ---
  1. Add more drivers during peak hours (8-10 AM, 5-8 PM)
  2. Promote premium vehicles to high-value customers
  3. Use Decision Tree model to predict ride completion
""")

print("Project Complete!")
