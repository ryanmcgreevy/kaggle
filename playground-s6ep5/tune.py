import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.utils import resample
from sklearn.preprocessing import OrdinalEncoder
import os
from sklearn.model_selection import cross_val_score
from dotenv import load_dotenv, dotenv_values 
import mlflow
import optuna
import sklearn
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.compose import ColumnTransformer
import argparse



def lgb_objective(trial, x, y, scoring, pipeline):
    # Setting nested=True will create a child run under the parent run.
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
        lgb_max_depth = trial.suggest_int("lgb_max_depth", 2, 10)
        lgb_n_estimators = trial.suggest_int("lgb_n_estimators", 100, 10000, step=100)
        lgb_learning_rate = trial.suggest_float("lgb_learning_rate", 0.01, 1.0)
        lgb_l2_regularization = trial.suggest_float("lgb_l2_regularization", 0.0, 100.0)
        lgb_max_bins = trial.suggest_int("lgb_max_bins", 2, 255)
        lgb_reg_alpha = trial.suggest_float("lgb_reg_alpha", 0.0, 100.0)
        lgb_min_data_in_leaf = trial.suggest_int("lgb_min_data_in_leaf", 1, 100)
        params = {
            "max_depth": lgb_max_depth,
            "class_weight": 'balanced',
            "n_estimators": lgb_n_estimators,
            "learning_rate": lgb_learning_rate,
            "reg_lambda": lgb_l2_regularization,
            "max_bin": lgb_max_bins,
            "reg_alpha": lgb_reg_alpha,
            "min_data_in_leaf": lgb_min_data_in_leaf,
            "objective": 'binary',
            'boosting_type': 'gbdt',
            #"device": 'cuda',
            "random_state": 1

        }
        # Log current trial's parameters
        mlflow.log_params(params)

        regressor_obj = LGBMClassifier(**params)
        pipeline.steps.append(['classifier', regressor_obj])

        scores = cross_val_score(pipeline, x, y, scoring=scoring, cv=5)
    
        pipeline.steps.pop()  # Remove the classifier step to avoid affecting subsequent trials

        # # Log current trial's error metric
        mlflow.log_metrics({"mean_score": np.mean(scores), "median_score": np.median(scores)})

        # Log the model file
        mlflow.sklearn.log_model(regressor_obj, name="model")
        # Make it easy to retrieve the best-performing child run later
        trial.set_user_attr("run_id", child_run.info.run_id)
        #return error
        return np.min([np.mean(scores), np.median(scores)])
    
def cb_objective(trial, x, y, scoring, pipeline):
    # Setting nested=True will create a child run under the parent run.
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
        cb_depth = trial.suggest_int("cb_depth", 2, 10)
        cb_iterations = trial.suggest_int("cb_iterations", 100, 10000, step=100)
        cb_learning_rate = trial.suggest_float("cb_learning_rate", 0.01, 1.0)
        cb_l2_leaf_reg = trial.suggest_float("cb_l2_leaf_reg", 0.0, 100.0)
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0)

        params = {
            "depth": cb_depth,
            "auto_class_weights": 'Balanced',
            "iterations": cb_iterations,
            "learning_rate": cb_learning_rate,
            "l2_leaf_reg": cb_l2_leaf_reg,
            'bagging_temperature': bagging_temperature,
            'loss_function': 'Logloss',
            'verbose': False,
            'task_type': 'GPU',
            #"device": 'cuda',
            "random_state": 1

        }
        # Log current trial's parameters
        mlflow.log_params(params)

        class CustomCatBoostClassifier(CatBoostClassifier):
            def __sklearn_clone__(self):
                return CustomCatBoostClassifier(**self.get_params())

        regressor_obj = CustomCatBoostClassifier(**params)
        pipeline.steps.append(['classifier', regressor_obj])

        scores = cross_val_score(pipeline, x, y, scoring=scoring, cv=5)

        pipeline.steps.pop()  # Remove the classifier step to avoid affecting subsequent trials

        # # Log current trial's error metric
        mlflow.log_metrics({"mean_score": np.mean(scores), "median_score": np.median(scores)})

        # Log the model file
        mlflow.sklearn.log_model(regressor_obj, name="model")
        # Make it easy to retrieve the best-performing child run later
        trial.set_user_attr("run_id", child_run.info.run_id)
        #return error
        return np.min([np.mean(scores), np.median(scores)])
    
def hb_objective(trial, x, y, scoring, pipeline):
    # Setting nested=True will create a child run under the parent run.
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
        hb_max_depth = trial.suggest_int("hb_max_depth", 2, 10)
        hb_max_iter = trial.suggest_int("hb_max_iter", 100, 10000, step=100)
        hb_learning_rate = trial.suggest_float("hb_learning_rate", 0.01, 1.0)
        hb_l2_regularization = trial.suggest_float("hb_l2_regularization", 0.0, 100.0)
        hb_max_bins = trial.suggest_int("hb_max_bins", 2, 255)
        params = {
            "max_depth": hb_max_depth,
            "class_weight": 'balanced',
            "max_iter": hb_max_iter,
            "learning_rate": hb_learning_rate,
            "l2_regularization": hb_l2_regularization,
            "max_bins": hb_max_bins,
            "loss": 'log_loss',
            "random_state": 1

        }
        # Log current trial's parameters
        mlflow.log_params(params)

        regressor_obj = HistGradientBoostingClassifier(**params)
        pipeline.steps.append(['classifier', regressor_obj])

        scores = cross_val_score(pipeline, x, y, scoring=scoring, cv=5)

        pipeline.steps.pop()  # Remove the classifier step to avoid affecting subsequent trials
        
        # # Log current trial's error metric
        mlflow.log_metrics({"mean_score": np.mean(scores), "median_score": np.median(scores)})

        # Log the model file
        mlflow.sklearn.log_model(regressor_obj, name="model")
        # Make it easy to retrieve the best-performing child run later
        trial.set_user_attr("run_id", child_run.info.run_id)

        #return error
        return np.min([np.mean(scores), np.median(scores)])
    
def run_optuna_study(objective_func, x, y, scoring, run_name, pipeline, n_trials=50):
    # Create a parent run that contains all child runs for different trials
    with mlflow.start_run(run_name=run_name) as run:
        # Log the experiment settings
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective_func(trial, x, y, scoring, pipeline), n_trials=n_trials)

        # Log the best trial and its run ID
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metrics({"best_error": study.best_value})
        if best_run_id := study.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id)

def process_data():
    df = pd.read_csv('./data/train.csv')
    X_full = df.drop(columns=['id', 'PitNextLap'])
    te_cols = ['Driver', 'Compound', 'Race', 'Year']
    sc_cols = X_full.drop(columns=te_cols).columns
    y = df['PitNextLap']

    preprocessor = ColumnTransformer(
        transformers=[
            ('te', TargetEncoder(categories='auto', target_type='binary', smooth='auto', cv=5, random_state=42), te_cols),
            ('sc', StandardScaler(), sc_cols)
        ]
    )

    pipe = Pipeline([('preprocessor', preprocessor)])
    return pipe, X_full, y


def main(args=None):

    # loading variables from .env file
    load_dotenv() 

    # Set up MLflow tracking
    mlflow.set_tracking_uri(os.getenv('MLFLOW_SERVER'))
    mlflow.set_experiment("S6E5: Hyperparameter Tuning Experiment")

    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    
    #read data and define pipeline
    pipe, X, y = process_data()

    print("Data loaded and pipeline defined. Starting hyperparameter tuning...")

    # Define the scoring metric
    scoring = make_scorer(roc_auc_score)

    match args.classifier:
        case 'lgbm':
            my_objective = lgb_objective
            classifier = "LightGBM"
        case 'cb':
            my_objective = cb_objective
            classifier = "CatBoost"
        case 'hgb':
            my_objective = hb_objective
            classifier = "HistGradientBoosting"
        case _:
            raise ValueError("Invalid classifier choice. Please choose from 'lgbm', 'cb', or 'hgb'.")
    
    run_optuna_study(my_objective, X, y, scoring, run_name=f"{classifier} Hyperparameter Tuning", pipeline=pipe, n_trials=50)

    # print("Running lightGBM hyperparameter tuning...")
    # # Run Optuna study for LightGBM
    # run_optuna_study(lgb_objective, X, y, scoring, run_name="LGBM Hyperparameter Tuning", pipeline=pipe, n_trials=50)

    # print("Running CatBoost hyperparameter tuning...")
    # # Run Optuna study for CatBoost
    # run_optuna_study(cb_objective, X, y, scoring, run_name="CatBoost Hyperparameter Tuning", pipeline=pipe, n_trials=50)

    # print("Running HistGradientBoosting hyperparameter tuning...")
    # # Run Optuna study for HistGradientBoosting
    # run_optuna_study(hb_objective, X, y, scoring, run_name="HistGradientBoosting Hyperparameter Tuning", pipeline=pipe, n_trials=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for LightGBM, CatBoost, and HistGradientBoosting using Optuna and MLflow.")
    parser.add_argument('--classifier', type=str, required=True, help="Choose the classifier to tune: 'lgbm', 'cb', or 'hgb'")
    args = parser.parse_args()
    main(args)