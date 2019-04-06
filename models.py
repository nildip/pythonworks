import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def randomforest_default(input_df, model_type, x_var, y_var, model_var_dict, rf_imp_thr=0.3):
    def rf_feat_importance(m, df):
        return pd.DataFrame({'variable':df.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False)
    # get user defined model params, revert to default if not found
    n_estimators = model_var_dict.get('n_estimators', [50, 100, 500, 1000])
    max_features = model_var_dict.get('max_features', ['sqrt', 'log2'])
    min_samples_leaf = model_var_dict.get('min_samples_leaf', [5, 20, 50])
    bootstrap = model_var_dict.get('bootstrap', [True, False])
    
    model_dict = {'RandomForestRegressor': RandomForestRegressor, 'RandomForestClassifier': RandomForestClassifier}
    X_train, X_val, Y_train, Y_val = train_test_split(input_df[x_var], input_df[y_var], random_state=666)
    model = model_dict[model_type]
    # 1st pass of model fititng
    # using grid search for estiamting optimal hyperparameters
    GridsearchCV = GridSearchCV(cv=5, estimator=model(n_jobs = -1), param_grid = {'n_estimators': n_estimators, 
                   'max_features': max_features, 'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}, verbose=0)
    GridsearchCV.fit(X_train, Y_train)
    best_params = GridsearchCV.best_params_
    # training model with optimal parameters
    model = model(n_estimators = best_params['n_estimators'], max_features = best_params['max_features'], 
                  min_samples_leaf = best_params['min_samples_leaf'], bootstrap = best_params['bootstrap'], 
                  verbose = 0, n_jobs = -1)
    model.fit(X_train, Y_train)
    print ("AUC - ROC : ", roc_auc_score(Y_val, model.predict(X_val)))
    # identify important features
    rf_imp_cols = input_df.columns
    rf_imp_cols = list(rf_feat_importance(model, input_df[x_var]).query("imp > @rf_imp_thr", 
                                                                        local_dict = {'rf_imp_thr': rf_imp_thr})['variable']) 
    # 2nd pass of model fitting with only selected important features to improve robustness
    model.fit(X_train[rf_imp_cols], X_train[y_var])
    print ("AUC - ROC : ", roc_auc_score(Y_val, model.predict(X_val)))
    # 3rd pass of model fitting with only selected important features on entire data
    model.fit(input_df[rf_imp_cols], input_df[y_var])    
    return model