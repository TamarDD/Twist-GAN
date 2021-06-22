# -*- coding: utf-8 -*-
"""# Utils"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from twistgan import *

from scipy.io.arff import loadarff 
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,accuracy_score

"""# Functions"""

def load_data_arff(file_path):
  """
  load .arff file type and encode text columns
  """
  raw_data = loadarff(file_path)
  df = pd.DataFrame(raw_data[0])
  str_df = df.select_dtypes([np.object]) 
  str_df = str_df.stack().str.decode('utf-8').unstack()
  df[str_df.columns] = str_df

  return df

def preprocess (df,test_size=0.3, seed=42):
  """
  The function prepares the dataset to random forest model.
  Operations:
    *normalize numric columns 
    *encode label column to numbers, 
    *spilt to train & test  
    *convert categories to one-hot matrix
  
  Args:
    df (DataFrame): tabular dataset
    test_size (float): represents the proportion of the dataset to include in the test split
    seed(int): controls the shuffling applied to the data before applying the split

  Returns:
    X_train, X_test, y_train, y_test


  """
  #cat_features = df.select_dtypes([np.object]).columns
  numeric_features = df.select_dtypes(include=np.number).columns
  df = normalize(df, numeric_features)
  
  y_col = df.columns[-1]
  if y_col not in numeric_features:
    from sklearn.preprocessing import LabelEncoder
    
    label_encoder = LabelEncoder()
    df[y_col] = label_encoder.fit_transform(df[y_col].values)
  
  X = df.iloc[:,:-1]
  y = df[y_col]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
  X_train, X_test = dummies(X_train, X_test)

  return X_train, X_test, y_train, y_test

def normalize(df, numeric_features):
  """
  Apply Min-Max scaling on dataset
  
  Args:
    df (DataFrame): tabular dataset
    numeric_features (list): numric columns names to apply normalization on
  
  return:
    normalized df
  """
  scaler = MinMaxScaler()
  for f in numeric_features:
    scaler = MinMaxScaler()
    df[f]= scaler.fit_transform(df[f].values.reshape(-1,1))

  return df

def dummies(train,test):
  """
    Apply one hot encoding to categorial features
    and remove categories in test that you did not have in training data
  """
  train= pd.get_dummies(train)
  test= pd.get_dummies(test)
  labels= sorted(list(set(train)))

  train, test = train.align(test, join='inner', axis=1)  # inner join

  return train,test

def run_rf(ds_name,df, save=False):
  """
  preprocess the dataset, train Random Forest model and evluate on testset
  
  Args:
    ds_name: the saving file name
    df (DataFrame): tabular dataset
    save (bool): indicates wheter to save the model or not
  
  Returns:
    acc(float): accuracy 
    roc_auc (float): auc metric
  """
  X_train, X_test, y_train, y_test = preprocess(df)
  clf = RandomForestClassifier(max_depth=20, random_state=0,n_estimators=100)
  clf.fit(X_train, y_train)
  y_pred=clf.predict(X_test)
  
  acc = round(accuracy_score(y_test,y_pred),2)
  
  probs = clf.predict_proba(X_test)[:,1]
  roc_auc = round(roc_auc_score(y_test, probs),2)
  if save:
    joblib.dump(clf, RF_MODEL_PATH+f"rf_model_{ds_name}.joblib")
  print(f'Accuracy: {acc}, AUC: {roc_auc}')
  return acc, roc_auc


def main():
  db_names = ['diabetes','german']
  path = [DIABETES_PATH,GERMAN_PATH]
  models = [DIABETES_RF_MODEL_PATH,GERMAN_RF_MODEL_PATH]

  for i in range(len(path)) :
    print(path[i])
    df = load_data_arff(path[i])
    run_rf(db_names[i],df,SAVE_MODEL) #train random forest on data and save model (opt)
    model = joblib.load(models[i])
    X_train, X_test, y_train, y_test = preprocess(df)
    df = pd.concat([X_train, X_test])
    gan = TwistGAN(df.shape[1],model)
    gan.train(df,EPOCHS)
    real_pred, fake_pred = gan.evalute_model(100)

DIABETES_PATH = '/dataset/diabetes.arff'
GERMAN_PATH = '/dataset/german_credit.arff'
RF_MODEL_PATH = '/models'
DIABETES_RF_MODEL_PATH = RF_MODEL_PATH+'rf_model_diabetes.joblib'
GERMAN_RF_MODEL_PATH = RF_MODEL_PATH+'rf_model_german.joblib'
EPOCHS = 5000
SAVE_MODEL = False
     


if __name__ == "__main__":
  
    main()
