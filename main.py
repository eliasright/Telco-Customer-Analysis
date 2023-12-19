import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import cross_val_score

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0').astype(float)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def split_data(df):
    X = df.drop(['Churn', 'gender', 'PhoneService'], axis=1)
    y = df['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=1221)

def build_preprocessor():
    categorical_features = ['SeniorCitizen', 'Partner', 'Dependents', 
                            'MultipleLines', 'InternetService', 'OnlineSecurity', 
                            'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                            'StreamingTV', 'StreamingMovies', 'Contract', 
                            'PaperlessBilling', 'PaymentMethod']
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

    return ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

def optimize_hyperparameters(X_train, y_train):
    def svm_evaluate(C, gamma):
        kernel = 'rbf'

        model = ImbPipeline(steps=[
            ('preprocessor', build_preprocessor()),  # Assuming build_preprocessor() is defined elsewhere
            ('smote', SMOTE(random_state=1221)),
            ('classifier', SVC(C=C, gamma=gamma, kernel=kernel, random_state=1221))
        ])
        f1 = cross_val_score(model, X_train, y_train, scoring='f1', cv=5).mean()
        return f1

    optimizer = BayesianOptimization(
        f=svm_evaluate,
        pbounds={
            'C': (0.01, 20),
            'gamma': (0.0001, 1),
        },
        random_state=1221
    )
    optimizer.maximize(init_points=10, n_iter=20)

    best_params = optimizer.max['params']
    best_params['kernel'] = 'rbf'
    return best_params

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

def main():
    file_path = 'Telco-Customer-Data.csv'
    df = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df)

    best_params = optimize_hyperparameters(X_train, y_train)
    print("Best parameters:", best_params)

    preprocessor = build_preprocessor()
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    model_best = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=1221)),
        ('classifier', SVC(**best_params, random_state=1221))
    ])
    model_best.fit(X_train_preprocessed, y_train)

    evaluate_model(model_best, X_test_preprocessed, y_test)

if __name__ == "__main__":
    main()