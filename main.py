import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from bayes_opt import BayesianOptimization
import shap

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0').astype(float)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def split_data(df):
    X = df.drop(['Churn', 'gender', 'PhoneService', 'customerID'], axis=1)
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

def optimize_hyperparameters(X_train, y_train, X_test, y_test):
    def rf_evaluate(n_estimators, max_depth):
        model = ImbPipeline(steps=[
            ('preprocessor', build_preprocessor()),
            ('smote', SMOTE(random_state=1221)),
            ('classifier', RandomForestClassifier(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                random_state=1221))
        ])
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)

    optimizer = BayesianOptimization(
        f=rf_evaluate,
        pbounds={'n_estimators': (10, 200), 'max_depth': (5, 30)},
        random_state=1221
    )
    optimizer.maximize(init_points=2, n_iter=10)
    return {k: int(v) for k, v in optimizer.max['params'].items()}

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

def main():
    file_path = 'Telco-Customer-Data.csv'
    df = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df)

    best_params = optimize_hyperparameters(X_train, y_train, X_test, y_test)
    print("Best parameters:", best_params)

    model_best = ImbPipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('smote', SMOTE(random_state=1221)),
        ('classifier', RandomForestClassifier(
            **best_params,
            random_state=1221))
    ])
    model_best.fit(X_train, y_train)
    evaluate_model(model_best, X_test, y_test)

if __name__ == "__main__":
    main()
