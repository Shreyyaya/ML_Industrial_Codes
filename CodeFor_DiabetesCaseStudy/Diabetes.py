###########################################################
# Required python pakages
###########################################################

import pandas as pd
import numpy as np

from sklearn.metrics import(
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import joblib


###########################################################
# File paths
###########################################################

INPUT = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
OUTPUT = "diabetes.csv"
MODEL_PATH = "Diabetes_pipeline.joblib"

###########################################################
#headers
###########################################################

HEADERS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]


###########################################################
# Function name : Read_data
# description : read data into pandas dataframe
# author : Shreya sandhanshive
# Date : 10 Aug,2025
###########################################################

def Read_data(path):
    # read data into pandas dataframe
    data = pd.read_csv(path, header = None)
    return data

###########################################################
# Function name : get_header()
# description : dataset headers
# input : dataset
# Output: header
# author : Shreya sandhanshive
# Date : 10 Aug,2025
###########################################################

def get_header(dataset):
    #return dataset headers
    return dataset.columns.values

###########################################################
# Function name : Add_header()
# description : add headers to dataset
# input : dataset
# Output : updated dataset with headers
# author : Shreya sandhanshive
# Date : 10 Aug,2025
###########################################################

def Add_header(dataset,headers):
    #add headers to dataset
    dataset.columns = headers
    return dataset

###########################################################
# Function name : data_file_to_csv
# Output : write data to csv
# author : Shreya sandhanshive
# Date : 10 Aug,2025
###########################################################

def data_file_to_csv():
    #converting raw data file to csv

    dataset = Read_data(INPUT)
    dataset = Add_header(dataset, HEADERS)
    dataset.to_csv(OUTPUT, index=False)
    print("File is saved succesfully!!")


###########################################################
# Function name : train_test_dataset()
# description :  split the dataset into training dataset and testing dataset    
# Input : dataset with information
# Output : dataset after splitting
# author : Shreya sandhanshive
# Date : 10 Aug,2025
###########################################################

def train_test_dataset(dataset, train_percentage,feature_headers,target_headers,random_state=42):
    #split the dataset into training dataset and testing dataset    
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[feature_headers], 
                                                        dataset[target_headers], train_size=train_percentage,
                                                        random_state=random_state, stratify=dataset[target_headers])
    return X_train, X_test, Y_train, Y_test


###########################################################
# Function name : dataset_stat()
# description : displays basic stat of dataset
# author : Shreya sandhanshive
# Date : 10 Aug,2025
###########################################################

def dataset_stat(dataset):
    #to print the stat of dataset
    print(dataset.describe(include = 'all'))


###########################################################
# Function name : build_pipeline()
# description : builds a pipeline
# author : Shreya sandhanshive
# Date : 10 Aug,2025
###########################################################

def build_pipeline():

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight = None
        ))
    ])
    return pipe


###########################################################
# Function name : train_pipeline()
# description : Trains the piepline
# author : Shreya sandhanshive
# Date : 10 Aug,2025
###########################################################

def train_pipeline(pipeline, X_train, Y_train):
    pipeline.fit(X_train, Y_train)
    return pipeline


###########################################################
# Function name : saving_model()
# description : saves the model
# author : Shreya sandhanshive
# Date : 10 Aug,2025
###########################################################

def saving_model(model, path = MODEL_PATH):
    joblib.dump(model,path)
    print("model saved to path: ", path)


###########################################################
# Function name : load_model()
# description : loads the trained model
# author : Shreya sandhanshive
# Date : 10 Aug,2025
###########################################################

def load_model(path = MODEL_PATH):
    model = joblib.load(path)
    print("Model loaded from:", path)
    return model    


###########################################################
# Function name : main
# description : Main function to start the execution of code
# author : Shreya sandhanshive
# Date : 10 Aug,2025
###########################################################

def main():


    #Ensure the csv exists
    data_file_to_csv()

    #Load the csv
    dataset = pd.read_csv(OUTPUT)

    #statistics of the dataset
    dataset_stat(dataset)

    #preparing features and target
    feature_headers = HEADERS[:-1]
    target_header = HEADERS[-1]

    #training
    X_train, X_test, Y_train, Y_test = train_test_dataset(dataset, 0.7, feature_headers, target_header)

    print("X_train shape", X_train.shape)
    print("X_test shape", X_test.shape)
    print("Y_train shape", Y_train.shape)
    print("Y_test shape", Y_test.shape)

    #buildinga nd training pipeline
    pipeline = build_pipeline()
    trained_model = train_pipeline(pipeline, X_train, Y_train)
    print("Trained pipeline : ", trained_model)

    #prediction
    prediction = trained_model.predict(X_test)

    #metrices
    print("Training accuracy", accuracy_score(Y_train, trained_model.predict(X_train)))
    print("Testing accuracy: ", accuracy_score(Y_test, prediction))
    print("Classification report:  \n", classification_report(Y_test,prediction))
    print("Confusion matrix:  \n", confusion_matrix(Y_test,prediction))

    #saving the model
    saving_model(trained_model, MODEL_PATH)

    #load the model and test a sample

    loaded = load_model(MODEL_PATH)
    sample = X_test.iloc[[0]]
    pred_loaded = loaded.predict(sample)
    print("Loaded model prediction for first test sample:", pred_loaded[0])




###########################################################
# application starter    
###########################################################

if __name__=="__main__":
    main()