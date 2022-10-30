''' Import packages and define functions '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from statsmodels import api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# Create function to fit linear regression model and visualize relationships
def show_plots(X, y):
    
    # Fit model, get fitted values, and calculate residual
    lm = LinearRegression()
    X = X.sort_values("date", ascending=True)
    lm.fit(X, y)
    y_hat = lm.predict(X)
    e = y - y_hat
    
    # Plot fitted values against residuals to detect non-constant variance
    plt.scatter(x=e, y=y_hat)
    plt.xlabel("residuals")
    plt.ylabel("fitted values")
    plt.show()
    
    # Plot residuals vs. date to see if there is correlation
    plt.plot(X["date"], e)
    plt.xlabel("date")
    plt.ylabel("residuals")
    plt.show()
    
    # Plot predictors against response to detect non-linear relationships
    for var in X.columns.to_list():
        plt.scatter(x=X[var].values, y=y)
        plt.xlabel(var)
        plt.ylabel("fitted values")
        plt.show()
        
# Create function for logging predictor
def log_predictor(df, var, zeros='increment'):
    if zeros == 'increment':
        df[var] = df[var].apply(lambda x: max(x, .01))
        df[f"log_{var}"] = np.log(df[var])
        df.drop(columns=[var], inplace=True)
    elif zeros == 'replace_w_nan':
        df[var].replace(0, np.nan, inplace=True)
        df[f"log_{var}"] = np.log(df[var])
        df.drop(columns=[var], inplace=True)
    else:
        raise Exception("How to deal with zeros is not specified.")

# Fit model and output summary
def fit_model(X, y):
    X_cnst = sm.add_constant(X)
    lm = sm.OLS(y, X_cnst).fit()
    print(lm.summary())

# Calculate VIF
def calculate_vif(df, threshold):
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif = vif.sort_values(by=["VIF"], ascending=False)
    high_vif_df = vif[vif["VIF"] > threshold]
    display(high_vif_df)
    
# Create interactions
def create_interactions(df, var1, var2):
    df[f"{var1}:{var2}"] = df[var1] * df[var2]

# Function for eliminating influential outliers
def elim_infl_outliers(data_label, X, y):
    # Fit model
    X_cnst = sm.add_constant(X)
    lm = sm.OLS(y, X).fit()
    
    # Get influence measures
    influence = lm.get_influence()
    
    # Obtain summary df of influence measures
    inf_df = influence.summary_frame()
    
    # Get leverage statistic
    leverage = influence.hat_matrix_diag
    
    # Plot leverage vs. studentized residuals
    plt.scatter(leverage, lm.resid_pearson)
    plt.title(f"Leverage vs. Studentized Residuals - {data_label} Dataset")
    plt.xlabel("Leverage")
    plt.ylabel("Studentized Residuals")
    
    # Set leverage threshold to (2k + 2)/n, where n is the number of observations and k is the number of features
    n = inf_df.shape[0]
    k = inf_df.shape[1]
    leverage_threshold = (2*k + 2)/n 
    
    # Calculate absolute value of studentized residuals
    inf_df["abs_student_resid"] = inf_df["student_resid"].apply(lambda x: abs(x))
    
    # Get indexes of influential outliers
    influential_idx = inf_df[(inf_df["abs_student_resid"] > 3)
                         & (inf_df["hat_diag"]) > leverage_threshold].index.tolist()
    
    print(f"The number of influential observations in the {data_label} dataset is {len(influential_idx)}.")
    
    # Drop observations from X and y that are influential outliers
    X.drop(influential_idx, inplace=True)
    y.drop(influential_idx, inplace=True)
    
def factorize_objects(df):
    ''' Convert series with object datatype into factors '''
    # Get all object variables
    obj_vars = df.select_dtypes(include=['object']).columns.tolist()
    
    for var in obj_vars:
        # Get variable values
        var_values = df[var].to_list()
        
        # Factorize variable values
        var_factors = pd.factorize(var_values)[0]
        
        # Drop the original variable column
        df.drop(columns=[var], inplace=True)
        
        # Create a new, factorized variable column of the same name
        df[var] = var_factors
      
def make_dates_useable(df):
    # Change date data types to int
    dt_vars = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    print(f"The datetime64[ns] variables in {df} are {dt_vars}")
    
    for var in dt_vars:
        df[var] = df[var].map(dt.datetime.toordinal)
        
def create_dummies(df, var, prefix=""):
    if prefix == "":
        prefix = var
    zip_dummies = pd.get_dummies(df[var], prefix=prefix, drop_first=True)
    zip_list = zip_dummies.columns.tolist()
    for col in zip_list:
        df[col] = zip_dummies[col]
    df.drop(columns=[var], inplace=True)
    
# def create_ordered_factors(df, var, order):
#     for i, item in enumerate(order):
#         df[var].replace(item, f"{i + 1}_{item}", inplace=True)
    
#     df[var].rename(str.lower, axis="columns", inplace=True)
#     df[var].columns.str.replace(' ', '_')
#     codes, uniques = pd.factorize(df[var], sort=True)
#     print(f"Codes: {codes}")
#     print(f"Unique values: {uniques}")
#     df[var] = codes
    
def plot_log_odds(X, y):
    
    # Add constant
    X_cnst = sm.add_constant(X)
    
    # Fit model and get predicted y values
    lm_num = sm.GLM(y, X_cnst, family=sm.families.Binomial(), missing="drop").fit()
    predicted = lm_num.predict(X_cnst)
    
    # Calculated log-odds and add it to X_num
    log_odds = np.log(predicted / (1 - predicted))
    
    # Create plots of each predictor vs. log-odds
    for var in X_cnst.columns.values:
        plt.scatter(x=X_cnst[var].values, y=log_odds)
        plt.xlabel(var)
        plt.ylabel("log-odds")
        plt.show()
        
    return lm_num, X_cnst
        
def get_model_summary(X, y, family=sm.families.Binomial(), missing="drop"):
    X_num = sm.add_constant(X)
    
    # Set up final dfs and fit model
    lm_final = sm.GLM(y, X_num, family=family, missing=missing).fit()
    
    print(lm_final.summary())
    
def plot_confusion_matrix(y_test, predictions):
    
    #Generate confusion matrix
    cf_matrix = confusion_matrix(y_test, predictions)
    
    # Create labels
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    # Plot confusion matrix
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    
    # Get accuracy score, recall, precision, and F1 score
    acc_score = "{0:.2%}".format(accuracy_score(y_test, predictions))
    recall = "{0:.2%}".format(recall_score(y_test, predictions))
    precision = "{0:.2%}".format(precision_score(y_test, predictions))
    f1 = "{0:.2%}".format(f1_score(y_test, predictions))
    print(f"Accuracy score: {acc_score}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")