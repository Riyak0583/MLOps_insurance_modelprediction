import pandas as pd

def data_ingestion():
    #Prepare Dockstring for the ingestion model
    '''
    * Project name: Insurance Price Prediction
    
    * Independent Columns {X}:'age', 'sex', 'bmi', 'children', 'smoker', 'region'

    * Dependent Columns {y}: 'charges'
    
    * Problem type: Supervised Machine Learning

    * ML task: Regression

    * Objective of the model : Predict the insurance price based on independent variables.
    '''

    # Load the data using pandas function
    df = pd.read_csv(r'C:\insurance_modelprediction\data\insurance.csv')

    #Drop the duplicate vlues before return the dataset
    df.drop_duplicates(inplace = True)

    return df