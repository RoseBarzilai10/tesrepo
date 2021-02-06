# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:44:40 2019

@author: Simone Spierings

Doel: Nieuwe klanten aan de zes gevonden clusters koppelen
Input: 
    - config.dictionary (P:\T-Mobile\_TMT\Scriptieonderzoek Simone - clusters\Data set)  -->  Data set waar de clusters op gebaseerd zijn (df_small_columns.csv bevat dezelfde inhoud)
    - Data set met nieuwe customers  -->  Format moet gelijk zijn aan format config.dictionary
        - TV_ACCESS: 1/yes, 0/no
        - MOBIEL: 1/yes, 0/no
        - AGE --> dummies, 5 groepen:
            - AGE_LOWER_30
            - AGE_30_40
            - AGE_40_50
            - AGE_50_60
            - AGE_HIGHER_60
        - LIFETIME --> dummies, 4 groepen, 3 kolommen in data (laatste groep wanneer 0 bij de eerste drie dummies):
            - LIFETIME_YRS_FIRST_YEAR: <= 12 maanden
            - LIFETIME_YRS_SECOND_YEAR: > 12 maanden en <= 24 maanden
            - LIFETIME_YRS_THIRD_YEAR: > 24 maanden en <= 36 maanden
            (- LIFETIME_YRS_MORE_YEAR: > 36 maanden) --> niet in inputdata
        - SPEED --> dummies, 2 groepen, 1 kolom in de data:
            - SPEED_LOW: <= 50
            (- SPEED_HIGH: >= 80) --> niet in inputdata
        - VOIP --> dummies, 2 groepen:
            - VOIP_START: Startpakket telefonie
            - VOIP_PLUS_EXTRA: Pluspakket of extra pakket telefonie
    
Output:
    - ID's nieuwe klanten met het bijbehorende cluster --> Python Dataframe & CSV met resultaten

Let op! Na bepaalde tijd kan beter een nieuwe clustering worden gemaakt op basis van nieuwe data


"""

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture 
import pickle
import cx_Oracle

# DIT DEEL INVULLEN
column_id = 'INVULLEN'    # Naam van de kolom met ids
schema_name = 'INVULLEN'  # Schema naam waar de data set met nieuwe klanten staat
password = 'INVULLEN'     # Wachtwoord van schema waar de data set met nieuwe klanten staat
db_name = 'INVULLEN'      # Database naam waar de data set met nieuwe klanten staat
table_name = 'INVULLEN'   # Naam van de tabel (data set met nieuwe klanten)


def main():
    # Load data from files
    train_df = load_train_data()
    test_df = load_test_data(schema_name, password, db_name, table_name)
    
    # Preprocessing test data
    test_id = get_id(test_df, column_id)
    test_df = drop_id(test_df, column_id)
    test_df = df_standardize(test_df)
    
    # Train & test model
    new_results = method_gmm(df_train=train_df, par_n_components=6, par_max_iter=1000, par_n_init=10, par_init_params='kmeans', par_random_state=1, df_test=test_df, df_test_id=test_id)
    
    return new_results

# Load train data 
def load_train_data():
    path = 'P:/T-Mobile/_TMT/Scriptieonderzoek Simone - clusters/Data set/config.dictionary'
    with open(path, 'rb') as config_dictionary_file:
        config_dictionary = pickle.load(config_dictionary_file)
    return config_dictionary  

# Load test data
def load_test_data(schema_name, password, db_name, table_name):
    c, conn = make_dbconnection(schema_name, password, db_name)
    df = load_data(conn, table_name)
    
    return df

# Get customer ids
def get_id(df, column_id):
    id_df = df.loc[:,column_id]
    return id_df

# Drop unique customer ids
def drop_id(df, column_id):
    df = df.drop([column_id], axis=1)
    return df 

# Standardize
def df_standardize(df):
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df))     
    scaled_df.columns = df.columns.values
    return scaled_df

# Train & test model
def method_gmm(df_train, par_n_components, par_max_iter, par_n_init, par_init_params, par_random_state, df_test, df_test_id):
    gmm = GaussianMixture(n_components=par_n_components,
                          max_iter = par_max_iter,
                          n_init = par_n_init,
                          init_params = par_init_params,
                          random_state = par_random_state).fit(df_train)
    labels = gmm.predict(df_test)
    gmm_clusters = pd.DataFrame(labels)
    gmm_clusters.columns = ['cluster']
    result = pd.concat([df_test_id, gmm_clusters], axis=1) 
    return result

# Make connection with database
def make_dbconnection(schema_name, password, db_name):
    connection_path = schema_name + '/' + password + '@' + db_name
    conn = cx_Oracle.connect(connection_path)
    c = conn.cursor()
    return c, conn

# Load data from database
def load_data(conn, table_name):
    sql_query = 'SELECT * FROM ' + table_name
    df = pd.read_sql(sql_query, con=conn)
    return df    


new_results = main()
new_results.to_csv('new_customers_with_cluster.csv')