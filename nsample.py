import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
data_set = pd.read_csv('Netflix Data new.csv')

# Display basic info about dataset
print("Dataset Information:\n", data_set.info())

# Pictographic analysis of missing values
msno.matrix(data_set)
plt.show()

# Extract independent variables (features) and dependent variable (target)
df_x = data_set.iloc[:, :-1]  # Keeps original column names
y = data_set.iloc[:, -1].values   # Last column (target)

# Handle missing values for numeric columns
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Identify categorical and numerical columns
categorical_cols = df_x.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df_x.select_dtypes(include=[np.number]).columns.tolist()

# Apply imputation to numeric columns only
df_x[numeric_cols] = imputer.fit_transform(df_x[numeric_cols])

# Apply OneHotEncoder to categorical columns
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)], remainder='passthrough')
x = ct.fit_transform(df_x)  # sparse_output=False ensures dense array

# Apply LabelEncoding to target variable (y)
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train[:, len(categorical_cols):] = sc.fit_transform(X_train[:, len(categorical_cols):])
X_test[:, len(categorical_cols):] = sc.transform(X_test[:, len(categorical_cols):])

# Pictographic Analysis
plt.figure(figsize=(10, 6))
sns.histplot(data_set['Release Year'].dropna(), bins=30, kde=True)
plt.title("Distribution of Release Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(y=data_set['Main Genre'], order=data_set['Main Genre'].value_counts().index)
plt.title("Distribution of Main Genre")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(data_set.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

print("Preprocessing completed successfully.")
