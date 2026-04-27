import pandas as pd
import numpy as np
from src.pipeline import train_model


#=======================================================
# Script for training the model and receiving outputs
#   Simply pass in your dataset
#========================================================

#-- 1) Define your dataset
df = pd.read_csv("toy_dataset.csv", index_col=0)

#-- 2) Define features
numerical_columns   = ['num1', 'num2']
categorical_columns = ['cat_var_1', 'cat_var_2']


#-- 3) Run
results_df = train_model(data = df, 
                         k=3, 
                         categorical_cols=categorical_columns, 
                         numerical_cols=numerical_columns, 
                         batch_size=100)

print(results_df)