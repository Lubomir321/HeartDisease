#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data and replace '?' with NaN
data = pd.read_csv("C:/Users/ACER/Desktop/domii2/heart-disease.csv", na_values='?')

# Print data types
print(data.dtypes)

# Check for missing values
print("Missing values per column before filling:")
print(data.isnull().sum())

# Encode all categorical columns
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col].astype(str))

# Check for missing values again
print("Missing values per column after encoding:")
print(data.isnull().sum())

# Handle missing values (fill with median)
data = data.fillna(data.median())

# Print numerical descriptions and category frequencies
print(data.describe())
for col in data.columns:
    if data[col].dtype == 'int64':
        print(data[col].value_counts())

# Check for missing values after handling them
print("Missing values per column after filling:")
print(data.isnull().sum())

# Visualizations
plt.scatter(data['age'], data['diameter narrowing'])
plt.xlabel('Age')
plt.ylabel('Diameter narrowing')
plt.title('Relationship between Age and Diameter Narrowing')
plt.show()

data['age'].hist()
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()

# Cholesterol vs Age with regression line
plt.scatter(data['age'], data['cholesterol'])
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Relationship between Age and Cholesterol')
z = np.polyfit(data['age'], data['cholesterol'], 1)
p = np.poly1d(z)
plt.plot(data['age'], p(data['age']), color='red')
plt.show()

# Max HR vs Age with regression line
plt.scatter(data['age'], data['max HR'])
plt.xlabel('Age')
plt.ylabel('Max HR')
plt.title('Relationship between Age and Max HR')
z = np.polyfit(data['age'], data['max HR'], 1)
p = np.poly1d(z)
plt.plot(data['age'], p(data['age']), color='red')
plt.show()

# Define the target variable
target = 'diameter narrowing'

# Extract features and target variable
X = data.drop(target, axis=1)
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and fit the scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Random Forest model on scaled data
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Importance
importances = model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print(feature_importances)

# Example 1
new_patient_data_1 = {
    'age': [60],
    'gender': [0],  # Female (0), Male (1)
    'chest pain': [1],  # Type 1 chest pain
    'rest SBP': [140],
    'cholesterol': [220],
    'fasting blood sugar > 120': [1],  # True (1) or False (0)
    'rest ECG': [0],  # Normal (0)
    'max HR': [160],
    'exerc ind ang': [1],  # Yes (1) or No (0)
    'ST by exercise': [3.0],
    'slope peak exc ST': [2],  # Upsloping (0), Flat (1), Downsloping (2)
    'major vessels colored': [1],
    'thal': [2]  # Fixed defect (0), Normal (1), Reversible defect (2)
}

# Example 2
new_patient_data_2 = {
    'age': [45],
    'gender': [1],  
    'chest pain': [0],  
    'rest SBP': [120],
    'cholesterol': [190],
    'fasting blood sugar > 120': [0],  
    'rest ECG': [2],  
    'max HR': [170],
    'exerc ind ang': [0],  
    'ST by exercise': [1.5],
    'slope peak exc ST': [1],  
    'major vessels colored': [0],
    'thal': [1]  
}

# Example 3
new_patient_data_3 = {
    'age': [55],
    'gender': [1],  
    'chest pain': [2],  
    'rest SBP': [130],
    'cholesterol': [200],
    'fasting blood sugar > 120': [0],  
    'rest ECG': [0],  
    'max HR': [150],
    'exerc ind ang': [1],  
    'ST by exercise': [2.0],
    'slope peak exc ST': [0],  
    'major vessels colored': [2],
    'thal': [0]  
}

# Example 4
new_patient_data_4 = {
    'age': [70],
    'gender': [0],  
    'chest pain': [1],  
    'rest SBP': [150],
    'cholesterol': [260],
    'fasting blood sugar > 120': [1],  
    'rest ECG': [2],  
    'max HR': [130],
    'exerc ind ang': [1],  
    'ST by exercise': [1.0],
    'slope peak exc ST': [1],  
    'major vessels colored': [3],
    'thal': [1]  
}

# Example 5
new_patient_data_5 = {
    'age': [50],
    'gender': [1],  
    'chest pain': [2],  
    'rest SBP': [140],
    'cholesterol': [230],
    'fasting blood sugar > 120': [0],  
    'rest ECG': [1],  
    'max HR': [160],
    'exerc ind ang': [0],  
    'ST by exercise': [2.5],
    'slope peak exc ST': [2],  
    'major vessels colored': [0],
    'thal': [2]  
}

# Example 6
new_patient_data_6 = {
    'age': [65],
    'gender': [1],  
    'chest pain': [0],  
    'rest SBP': [130],
    'cholesterol': [180],
    'fasting blood sugar > 120': [0],  
    'rest ECG': [1],  
    'max HR': [150],
    'exerc ind ang': [0],  
    'ST by exercise': [1.0],
    'slope peak exc ST': [1],  
    'major vessels colored': [0],
    'thal': [0]  
}

# Example 7
new_patient_data_7 = {
    'age': [58],
    'gender': [0],  
    'chest pain': [2],  
    'rest SBP': [150],
    'cholesterol': [240],
    'fasting blood sugar > 120': [1],  
    'rest ECG': [0],  
    'max HR': [170],
    'exerc ind ang': [1],  
    'ST by exercise': [2.0],
    'slope peak exc ST': [0],  
    'major vessels colored': [1],
    'thal': [1]  
}

# Example 8
new_patient_data_8 = {
    'age': [48],
    'gender': [1],  
    'chest pain': [0],  
    'rest SBP': [120],
    'cholesterol': [210],
    'fasting blood sugar > 120': [0],  
    'rest ECG': [2],  
    'max HR': [160],
    'exerc ind ang': [0],  
    'ST by exercise': [1.5],
    'slope peak exc ST': [2],  
    'major vessels colored': [0],
    'thal': [2]  
}

# Example 9
new_patient_data_9 = {
    'age': [55],
    'gender': [1],  
    'chest pain': [1],  
    'rest SBP': [140],
    'cholesterol': [190],
    'fasting blood sugar > 120': [1],  
    'rest ECG': [1],  
    'max HR': [170],
    'exerc ind ang': [0],  
    'ST by exercise': [1.0],
    'slope peak exc ST': [1],  
    'major vessels colored': [0],
    'thal': [0]  
}

# Example 10
new_patient_data_10 = {
    'age': [62],
    'gender': [0],  
    'chest pain': [2],  
    'rest SBP': [130],
    'cholesterol': [200],
    'fasting blood sugar > 120': [0],  
    'rest ECG': [0],  
    'max HR': [150],
    'exerc ind ang': [1],  
    'ST by exercise': [2],
    'slope peak exc ST': [2],  
    'major vessels colored': [2],
    'thal': [1]  
}

# Example 11
new_patient_data_11 = {
    'age': [57],
    'gender': [1],  
    'chest pain': [1],  
    'rest SBP': [150],
    'cholesterol': [220],
    'fasting blood sugar > 120': [1],  
    'rest ECG': [0],  
    'max HR': [140],
    'exerc ind ang': [1],  
    'ST by exercise': [2.5],
    'slope peak exc ST': [1],  
    'major vessels colored': [3],
    'thal': [2]  
}

# Example 12
new_patient_data_12 = {
    'age': [40],
    'gender': [1],  
    'chest pain': [0],  
    'rest SBP': [120],
    'cholesterol': [180],
    'fasting blood sugar > 120': [0],  
    'rest ECG': [0],  
    'max HR': [160],
    'exerc ind ang': [0],  
    'ST by exercise': [0.5],
    'slope peak exc ST': [0],  
    'major vessels colored': [0],
    'thal': [1]  
}

# Example 13
new_patient_data_13 = {
    'age': [70],
    'gender': [0],  
    'chest pain': [2],  
    'rest SBP': [160],
    'cholesterol': [250],
    'fasting blood sugar > 120': [1],  
    'rest ECG': [2],  
    'max HR': [130],
    'exerc ind ang': [1],  
    'ST by exercise': [2.0],
    'slope peak exc ST': [2],  
    'major vessels colored': [2],
    'thal': [1]  
}

# Example 14
new_patient_data_14 = {
    'age': [35],
    'gender': [1],  
    'chest pain': [0],  
    'rest SBP': [110],
    'cholesterol': [170],
    'fasting blood sugar > 120': [0],  
    'rest ECG': [0],  
    'max HR': [150],
    'exerc ind ang': [0],  
    'ST by exercise': [0.0],
    'slope peak exc ST': [0],  
    'major vessels colored': [0],
    'thal': [2]  
}

# Example 15
new_patient_data_15 = {
    'age': [68],
    'gender': [0],  
    'chest pain': [2],  
    'rest SBP': [145],
    'cholesterol': [230],
    'fasting blood sugar > 120': [1],  
    'rest ECG': [2],  
    'max HR': [160],
    'exerc ind ang': [1],  
    'ST by exercise': [1.5],
    'slope peak exc ST': [1],  
    'major vessels colored': [1],
    'thal': [1]  
}
new_patients_data = [new_patient_data_1, new_patient_data_2, new_patient_data_3,
                     new_patient_data_4, new_patient_data_5, new_patient_data_6,
                     new_patient_data_7, new_patient_data_8, new_patient_data_9,
                     new_patient_data_10, new_patient_data_11, new_patient_data_12,
                     new_patient_data_13, new_patient_data_14, new_patient_data_15]

# Iterate over each new patient data and make predictions
for i, new_patient_data in enumerate(new_patients_data, start=1):
    # Convert to DataFrame
    new_patient_df = pd.DataFrame(new_patient_data, index=[0])  # Ensure the data is in the shape of (1, n_features)
    
    # Transform the new patient data
    new_patient_scaled = scaler.transform(new_patient_df)
    
    # Make prediction for the new patient
    new_prediction = model.predict(new_patient_scaled)
    
    # Output the prediction
    print(f"Predicted class for new patient {i}: {new_prediction[0]}")

# Example new patient data
new_patient_data_unhealthy = {
    'age': 65,
    'gender': 1,  
    'chest pain': 2,  
    'rest SBP': 170,
    'cholesterol': 280,
    'fasting blood sugar > 120': 1,  
    'rest ECG': 2,  
    'max HR': 120,
    'exerc ind ang': 1,  
    'ST by exercise': 3.5,
    'slope peak exc ST': 2,  
    'major vessels colored': 3,
    'thal': 2  
}

# Convert new patient data to DataFrame
new_patient_unhealthy = pd.DataFrame(new_patient_data_unhealthy, index=[0])

# Transform the new patient data using the fitted scaler
new_patient_scaled_unhealthy = scaler.transform(new_patient_unhealthy)

# Make prediction for the new patient
new_prediction_unhealthy = model.predict(new_patient_scaled_unhealthy)
print("Predicted class for the new unhealthy patient:", new_prediction_unhealthy[0])


# In[ ]:




