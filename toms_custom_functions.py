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
    """
    Description
    -----------
    1) Fits a regression model with Scikit-Learn
    2) Creates a scatterplot of residuals against fitted values for a training set
    3) Creates a scatterplot of a field called date against residuals for a training set
    4) Creates a scatterplot of fitted values against each predictor variable
    
    Parameters
    ----------
    X : pandas.DataFrame
        A design matrix
    y : pandas.Series
        A vector of true values for the response variable
    """
    
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
    """
    Description
    -----------
    Deals with zero values, logs the variable, and drops the original column
    
    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing the variable you want to log
    var : str
        The variable you want to log
    zeros : str
        Indicates how to deal with zeros, by either incrementing them by .01 or replacing them with NaN (default is 'increment')
        
    Raises
    ------
    Exception
        If a method (increment or replace_w_nan) is not specified for how to deal with zero values, an exception will be thrown
    """
    
    if zeros == 'increment':
        df[var] = df[var].apply(lambda x: max(x, .01))
    elif zeros == 'replace_w_nan':
        df[var].replace(0, np.nan, inplace=True)
    else:
        raise Exception("How to deal with zeros is not specified.")
    
    df[f"log_{var}"] = np.log(df[var])
    df.drop(columns=[var], inplace=True)

# Fit model and output summary
def fit_model(X, y):
    """
    Parameters
    ----------
    X : pandas.DataFrame
        A design matrix
    y : pandas.Series
        A vector of true values for the response variable
    """
    
    X_cnst = sm.add_constant(X)
    lm = sm.OLS(y, X_cnst).fit()
    print(lm.summary())

# Calculate VIF
def calculate_vif(df, threshold):
    """
    Description
    -----------
    1) Calculate variable inflation factor on fields in a dataframe
    2) Sort them in descending order
    3) Filter the dataframe of VIF values for ones higher than the specified threshold
    
    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of values you want to calculate VIF for
    threshold : int
        An integer specifying the non-inclusive lower bound, all values above which are considered high VIF
    """
    
    vif = pd.DataFrame()
    vif["variables"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif = vif.sort_values(by=["VIF"], ascending=False)
    high_vif_df = vif[vif["VIF"] > threshold]
    display(high_vif_df)
    
# Create interactions
def create_interactions(df, var1, var2):
    """
    Description
    -----------
    Create an interaction variable from two pre-existing variables
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the variables you want to create an interaction out of
    var1 : pandas.Series
        One variable in the interaction
    var2 : pandas.Series
        The other variable in the interaction
    """
    
    df[f"{var1}:{var2}"] = df[var1] * df[var2]

# Function for eliminating influence outliers
def elim_infl_outliers(data_label, X, y):
    """
    Description
    -----------
    1) Fit an OLS model
    2) Get dataframe of influential measures
    3) Identify and remove influential outliers
    
    Parameters
    ----------
    data_label : str
        The label describing for which dataset influential outliers are being eliminated
    X : pandas.Series
        A design matrix
    y : pandas.Series
        A vector of true values for the response variable
    """
    
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
    """
    Description
    -----------
    Turn dataframe columns of type object into factors (does not specify order)
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the columns you want factorized
    """
    
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
    """
    Description
    -----------
    Turns dataframe columns of object type 'datetime64[ns]' into ordinal columns
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the date columns you want converted to ordinal
    """
    
    # Change date data types to int
    dt_vars = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    print(f"The datetime64[ns] variables in {df} are {dt_vars}")
    
    for var in dt_vars:
        df[var] = df[var].map(dt.datetime.toordinal)
        
def create_dummies(df, var, prefix=""):
    """
    Description
    -----------
    Creates dummy variables from a specified column, then drops the original column
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the columns you want to create dummies out of
    var : str
        The name of the column you want to create dummy variables for
    prefix : str
        The prefix you want to append to all dummy variables created in this function call (defaults to blank)
    """
    
    if prefix == "":
        prefix = var
    zip_dummies = pd.get_dummies(df[var], prefix=prefix, drop_first=True)
    zip_list = zip_dummies.columns.tolist()
    for col in zip_list:
        df[col] = zip_dummies[col]
    df.drop(columns=[var], inplace=True)
    
def plot_log_odds(X, y):
    """
    Description
    -----------
    1) Fit a logistic regression model
    2) Get fitted values
    3) Calculate log-odds
    4) Plot each variable against log-odds
    
    Parameters
    ----------
    X : pandas.DataFrame
        A design matrix
    y : pandas.Series
        A vector of true values for a response variable
        
    Returns
    -------
    lm_num
        Fitted model from sm.GLM
    X_cnst
        The design matrix, with intercept added, from the model
    """
    
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
    """
    Description
    -----------
    Fit a model using sm.GLM and get its summary output
    
    Parameters
    ----------
    X : pandas.DataFrame
        A design matrix
    y : pandas.Series
        A vector of true values for a response variable
    family : statsmodels.genmod.families.family.
        A model family from the statsmodels package (defaults to sm.families.Binomial())
    missing : str
        Indicates whether to drop missing values when fitting the model (defaults to "drop")
    """
    
    X_num = sm.add_constant(X)
    
    # Set up final dfs and fit model
    lm_final = sm.GLM(y, X_num, family=family, missing=missing).fit()
    
    print(lm_final.summary())
    
def plot_confusion_matrix(y_test, predictions):
    """
    Description
    -----------
    1) Generate, format, and output confusion matrix
    2) Report on performance measures
    
    Parameters
    ----------
    y_test : pandas.Series
        A vector of true values for a response variable in a test set
    predictions : pandas.Series
        A vector of fitted values for a response variable in a test set
    """
    
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