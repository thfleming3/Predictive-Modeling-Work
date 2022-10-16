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
def log_predictor(df, var):
    df[var] = df[var].apply(lambda x: max(x, .01))
    df[f"log_{var}"] = np.log(df[var])
    df.drop(columns=[var], inplace=True)

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