Titanic - Machine Learning Disaster

This repository contains a machine learning project aimed at predicting the survival of passengers aboard the Titanic, using the popular Kaggle dataset. The project walks through various stages of data analysis and machine learning model building, from data preprocessing to model evaluation.
Dataset

The dataset includes features like:

    PassengerId: Unique ID for each passenger
    Pclass: Ticket class (1st, 2nd, 3rd)
    Name: Name of the passenger
    Sex: Gender of the passenger
    Age: Age of the passenger
    SibSp: Number of siblings/spouses aboard
    Parch: Number of parents/children aboard
    Ticket: Ticket number
    Fare: Fare paid for the ticket
    Cabin: Cabin number
    Embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

Project Structure

    Data Preprocessing:
        Handling missing data, especially in columns like Age and Embarked.
        Feature engineering to create new relevant features.
        Encoding categorical variables, such as Sex and Embarked, for use in machine learning models.

    Exploratory Data Analysis (EDA):
        Visualizing survival rates based on different features like Sex, Pclass, Fare, and more.
        Examining the relationships between independent variables and the target variable (survival).

    Modeling:
        Building models using various machine learning algorithms, including:
            Logistic Regression
            Decision Trees
            Random Forest Classifier
            Gradient Boosting Classifier
        Hyperparameter tuning and cross-validation to improve model performance.

    Model Evaluation:
        Using metrics like accuracy, precision, recall, F1-score, and ROC-AUC to evaluate the models.
        Plotting ROC curves to visualize model performance.

    Submission:
        Generating predictions on the test set and preparing a CSV file for Kaggle submission.

Results

    The notebook provides a detailed breakdown of each model's performance, focusing on maximizing accuracy and robustness in predictions.
    The project highlights the use of ROC curves and AUC scores for model evaluation.

Dependencies

The following Python libraries are required:

    Pandas
    NumPy
    Matplotlib
    Seaborn
    Scikit-learn

You can install the dependencies using:

bash

pip install -r requirements.txt

How to Run

    Clone the repository:

    bash

git clone https://github.com/your-username/titanic-ml.git
cd titanic-ml

Install dependencies:

bash

pip install -r requirements.txt

Open and run the Jupyter Notebook:

bash

jupyter notebook main.ipynb

Explore the data and modify the notebook to experiment with different models and features.
