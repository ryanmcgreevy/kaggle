# Playground Series S6E5: Predicting F1 Pit Stops

[Competition link](https://www.kaggle.com/competitions/playground-series-s6e5/overview)

## Overview
This repository contains my work for Kaggle Playground Series Season 6 Episode 5.

The primary focus was building a complete competition workflow: exploratory analysis, preprocessing, model training, ensembling, hyperparameter tuning, experiment tracking, and submission generation.

## Project Goals
- Build a repeatable end-to-end Kaggle workflow for binary classification
- Compare gradient boosting models and a custom PyTorch neural network
- Improve model quality through Optuna-based hyperparameter tuning
- Track and organize experiments with MLflow
- Run larger tuning jobs remotely on AWS/SageMaker with a custom Docker image

## Where to Start
The best entry point is:
- [submission.ipynb](submission.ipynb)

This is the final Kaggle workflow notebook. It includes:
- quick EDA and target balance checks
- preprocessing with target encoding + feature scaling
- model setup for LightGBM, CatBoost, HistGradientBoosting, and a custom NN wrapper
- stacking ensemble training with logistic regression as the meta-model
- test inference and creation of the final submission CSV

## Tuning and Experiment Tracking
Hyperparameter tuning and experiment logging are handled in:
- [tune.py](tune.py)

Key details:
- Optuna is used to optimize model hyperparameters across multiple model families.
- Each Optuna trial is logged as a nested MLflow run.
- Trial parameters and cross-validation metrics (mean/median roc_auc) are logged to MLflow.
- The best trial parameters are promoted and recorded in the parent run.
- Training/evaluation uses 5-fold cross-validation with roc_auc scoring.

## AWS Remote Tuning Workflow
The AWS launcher script is:
- [aws.py](aws.py)

This script uses SageMaker Remote Functions to run tuning jobs in AWS and is configured to execute [tune.py](tune.py) remotely.

Container/runtime notes:
- The runtime image is pulled from ECR via the configured image_uri in [aws.py](aws.py).
- The image is built from the local [Dockerfile](Dockerfile).
- Python dependencies are installed from [requirements.txt](requirements.txt).
- The script sets the MLflow tracking server ARN through the MLFLOW_SERVER environment variable before invoking tuning.

## Other Notebooks
- [explore.ipynb](explore.ipynb): exploratory analysis and feature-level iteration
- [clean.ipynb](clean.ipynb): intermediate workflow cleanup and experimentation
- [nn.ipynb](nn.ipynb): custom neural-network focused experiments

## Results
- Final submission workflow is centered in [submission.ipynb](submission.ipynb).
- Core ensemble approach: stacking gradient boosting models plus a custom NN model.
- Public leaderboard score from the notebook workflow: **0.94896**.

## Repository Notes
This project includes both final and exploratory work. Some scripts and notebooks overlap intentionally so experiments, tuning runs, and modeling iterations are preserved for reference.

## Next Steps
- Extend Optuna searches (more trials and conditional search spaces)
- Add richer feature engineering around tyre-life progression and race context
- Compare stacking against weighted voting and blended out-of-fold strategies
- Standardize training and inference scripts for easier reruns
