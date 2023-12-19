import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Telco-Customer-Data.csv'
df = pd.read_csv(file_path)

# TotalCharges is '' due to tenure 0. Replace with 0
df.replace({'TotalCharges': {'': 0}}, inplace=True)

# Convert Churn into a binary variable
df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)

# Convert SeniorCitizen into a categorical variable
df['SeniorCitizen'] = df['SeniorCitizen'].astype('category')

# Check for null values after replacing empty strings
null_values = df.isnull().sum()
print(f"Null Values:\n{null_values}")

# Handling null values (if necessary)
# You might need to handle the null values here depending on the output of the previous step
# For example, using fillna() or dropna()

# Check for duplicates
duplicates = df.duplicated().sum()
print("\nNumber of Duplicate Rows:", duplicates)

# Split dataset into target, continuous, and categorical variables
col_target = 'Churn'

cols_continuous = ['tenure', 'MonthlyCharges', 'TotalCharges']
df_continuous = df[cols_continuous]

# Convert continuous columns to numeric, handling non-numeric values
df_continuous = df_continuous.apply(pd.to_numeric, errors='coerce')

# Check for any additional null values created by coercion
print("\nNew Null Values in Continuous Columns:")
print(df_continuous.isnull().sum())

# Summary statistics for continuous features
summary_statistics = df_continuous.describe()
print("\nSummary Statistics")
print(summary_statistics.transpose())

# Unique values for categorical features
print("\nCategorical Features")
cols_categorical = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                    'Contract', 'PaperlessBilling', 'PaymentMethod']
for col in cols_categorical + [col_target]:
    unique_values = df[col].unique()
    print(f"Unique Values '{col}' (total {len(unique_values)}): {unique_values}")

# Boxplot for continuous variables
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axes.flatten()):
    sns.boxplot(x=df_continuous[cols_continuous[i]], ax=ax)
    ax.set_title(cols_continuous[i], fontsize=10)
    ax.set_xlabel('')
fig.subplots_adjust(hspace=0.3, wspace=0.1)  
plt.show()

# Pearson correlation
pearson_corr = pd.concat([df_continuous, df[col_target]], axis=1).corr()
plt.figure(figsize=(8, 4))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Pearson Correlation Matrix')
plt.subplots_adjust(bottom=0.3)
plt.show()

# Chi-squared test for categorical features
df_chi = df[cols_categorical + [col_target]].astype('category')
chi_squared_p_values = pd.DataFrame(index=cols_categorical, columns=[col_target], dtype=float)
for col in cols_categorical:
    confusion_matrix = pd.crosstab(df_chi[col], df_chi[col_target])
    # Check for expected frequencies condition
    if not np.all(confusion_matrix.values.flatten() >= 5):
        print(f"\nFailed to meet min 5 rule assumption: {col}")
        print(confusion_matrix.values)
    chi2, p, _, _ = chi2_contingency(confusion_matrix)
    chi_squared_p_values.loc[col, col_target] = p

plt.figure(figsize=(10, 8))
sns.heatmap(chi_squared_p_values, annot=True, cmap='coolwarm_r', fmt='.2e')
plt.title("Chi-squared Test Results for Categorical Features against Churn")
plt.subplots_adjust(left=0.3)
plt.show()
