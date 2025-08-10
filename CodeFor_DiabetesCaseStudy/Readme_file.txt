----------------------------------------------------Diabetes Prediction using Random Forest Pipeline-------------------------------------------------------------


OVERVIEW: 
This project builds a machine learning pipeline to predict the likelihood of diabetes using the Diabetes Dataset. 

Working of the pipeline:
    Reads raw data from an online source
    Cleans and prepares the dataset
    Splits data into training and testing sets
    Trains a Random Forest model with preprocessing
    Evaluates performance using accuracy, classification report, and confusion matrix
    Saves and loads the trained model for reuse


DATASET:
    The dataset used is the Pima Indians Diabetes Dataset from:
        https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv


FEATURES OF DATASET:
    Pregnancies, Glucose, BloodPressure , SkinThickness , Insulin, BMI, DiabetesPedigreeFunction, Age


TARGET: 
    Outcome (0 = No Diabetes, 1 = Diabetes)


BASIC REQUIREMENTS:

    Install required packages:
        pip install pandas numpy scikit-learn joblib



STRUCTURE OF FILE: 

        diabetes.csv                 # Processed dataset (created during execution)
        Diabetes_pipeline.joblib     # Saved trained model (created during execution)
        Diabetes.py                  # Main Python script
        README.md                    # Project documentation


PYTHON VERSION: :
    Ensure Python 3.x is installed.


INSTALLATION DEPENDENCIES:

        pip install pandas numpy scikit-learn joblib


COMMAND TO RUN THE SCRIPT:
        python Diabetes.py


OUTPUT OF THE PYTHON SCRIPT
        Print dataset statistics.
        Display training/testing dataset shapes.
        Show model training and testing accuracy.
        Print classification report and confusion matrix.
        Save the trained model to Diabetes_pipeline.joblib.
        Load the saved model and predict for a sample test case.