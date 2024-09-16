import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Load data
data = pd.read_csv('train.csv')

# Preprocess data
data['Cabin'] = data['Cabin'].fillna('N')
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Cabin'] = data['Cabin'].apply(lambda x: x[0])
data['AgeGroup'] = pd.cut(data['Age'], bins=[0,18,60,100], labels=['Kids','Adult','Old'])
data['Survived'] = data['Survived'].map({0: 'Died', 1: 'Survived'})
data['CabinLabel'] = LabelEncoder().fit_transform(data['Cabin'])
data['SexLabel'] = LabelEncoder().fit_transform(data['Sex'])

# Prepare the data
X = data[['Pclass', 'Age', 'Fare', 'CabinLabel', 'SexLabel']]
y = data['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the models
model_lr = LogisticRegression(max_iter=200)
model_random_forest = RandomForestClassifier(n_estimators=100)
model_lr.fit(X_train, y_train)
model_random_forest.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = model_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
y_pred_rf = model_random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Streamlit app
st.title('Titanic Survival Prediction')

# Show dataset
st.header('Dataset')
st.write(data.head(10))

# Show predictions
st.header('Predictions')
st.write(f'Logistic Regression Accuracy: {accuracy_lr:.2f}')
st.write(f'Random Forest Accuracy: {accuracy_rf:.2f}')

# Confusion Matrix
st.header('Confusion Matrix')
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', ax=ax[0])
ax[0].set_title('Logistic Regression')
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', ax=ax[1])
ax[1].set_title('Random Forest')
st.pyplot(fig)

# ROC Curve
st.header('ROC Curve')
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Logistic Regression ROC
fpr_lr, tpr_lr, _ = roc_curve(LabelEncoder().fit_transform(y_test), LabelEncoder().transform(y_pred_lr), pos_label=1)
roc_auc_lr = auc(fpr_lr, tpr_lr)
ax[0].plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_lr:.2f})')
ax[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax[0].set_xlim([0.0, 1.0])
ax[0].set_ylim([0.0, 1.05])
ax[0].set_xlabel('False Positive Rate')
ax[0].set_ylabel('True Positive Rate')
ax[0].set_title('Logistic Regression ROC')
ax[0].legend(loc="lower right")

# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(LabelEncoder().fit_transform(y_test), LabelEncoder().transform(y_pred_rf), pos_label=1)
roc_auc_rf = auc(fpr_rf, tpr_rf)
ax[1].plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_rf:.2f})')
ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([0.0, 1.05])
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('Random Forest ROC')
ax[1].legend(loc="lower right")

st.pyplot(fig)