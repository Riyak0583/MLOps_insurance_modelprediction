from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd

def model_build(X_train,X_test,y_train,y_test):
   
    models = {'DT':DecisionTreeRegressor(),
              'RF':RandomForestRegressor(),
              'GB':GradientBoostingRegressor(),
              'XGB':XGBRegressor(),
              'LGBM':LGBMRegressor()
       
    }
   
    error = {}
    score = {}
   
    for model_name,model in models.items():
        model.fit(X_train,y_train)  # Seen Data
        y_pred = model.predict(X_test) # Unseen Data
       
        mse = mean_squared_error(y_test,y_pred)
        r2 = r2_score(y_test,y_pred)
       
        error[model_name] = mse
        score[model_name] = r2
       
    # Optional: Combine into a DataFrame for easy comparison
    results_df = pd.DataFrame({
        'MSE': error,
        'R2_Score': score
    }).sort_values(by='R2_Score', ascending=False)
   
    return results_df, error, score    