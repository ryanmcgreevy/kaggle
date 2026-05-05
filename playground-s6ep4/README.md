# Playground Series S6E4: Predicting Irrigation Need

[Competition link](https://www.kaggle.com/competitions/playground-series-s6e4/overview)

## Overview
This repository contains my first Kaggle competition in over a decade and marks my return to data science and machine learning work.

The main objective was learning: reacquainting myself with the Kaggle platform and practicing modern ML workflows, tools, and frameworks.

## Project Goals
- Build end-to-end competition workflow habits and familiarity
- Practice EDA, feature engineering, model training, and ensembling
- Explore AWS/SageMaker-based workflow experimentation

## Where to Start
The best entry point is:
- [submission.ipynb](submission.ipynb)

This notebook presents the cleaned final pipeline and summary of the approach.
Public Kaggle notebook: [PS-S6EP4 Basic EDA and Gradient Boost Ensemble](https://www.kaggle.com/code/ryanmcgreevy/ps-s6ep4-basic-eda-and-gradient-boost-ensemble)

## Other Notebooks
- [nn_knb.ipynb](nn_knb.ipynb): PyTorch neural network experiment (learning-focused, not part of final ensemble. It was never tuned well enough to include.)
- [aws-step-pipe.ipynb](aws-step-pipe.ipynb): AWS workflow tests
- [aws-test.ipynb](aws-test.ipynb): additional AWS/SageMaker experimentation

## Results
- Final model: soft-voting ensemble of three tuned gradient boosting classifiers: LightGBM, CatBoost, and HistGradientBoostingClassifier.
- Tuning workflow: Optuna + MLflow with 5-fold cross-validation (as summarized in [submission.ipynb](submission.ipynb)).
- Class imbalance handling: class weighting/balancing options enabled across the gradient boosting models.
- Inference pipeline: one-hot encode test features to match training columns, predict with the ensemble, inverse-transform encoded labels, and write submission.csv.
- Public leaderboard score: **0.97031**.

## Repository Notes
Because this project was primarily for learning, the repository intentionally preserves exploratory and partially disjointed work for future reference.

## Next Steps
- Improve neural network tuning and compare against boosting models
- Refine ensembling strategy
- Standardize training/evaluation scripts for reuse in future competitions

