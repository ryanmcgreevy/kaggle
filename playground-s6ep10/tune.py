import argparse
import json

import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, TargetEncoder


def make_pipeline(X, classifier_params, random_state):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(include="number").columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                TargetEncoder(
                    target_type="binary",
                    smooth="auto",
                    cv=5,
                    random_state=random_state,
                ),
                categorical_cols,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("scaler", RobustScaler()),
                        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
                    ]
                ),
                numeric_cols,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    **classifier_params,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Tune a HistGradientBoostingClassifier for S6E10 using cross-validated ROC AUC."
    )
    parser.add_argument("--data", default="data/train.csv", help="Training CSV path")
    parser.add_argument("--target", default="satisfaction", help="Target column")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of stratified CV folds")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50_000,
        help="Stratified sample size for tuning; use 0 for the full dataset",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.n_trials < 1:
        parser.error("--n-trials must be at least 1")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    if args.max_samples < 0:
        parser.error("--max-samples must be nonnegative")

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column {args.target!r} was not found in {args.data!r}")

    X = df.drop(columns=["id", args.target], errors="ignore")
    y = df[args.target]

    if 0 < args.max_samples < len(X):
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=args.max_samples,
            random_state=args.random_state,
            stratify=y,
        )

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    def objective(trial):
        classifier_params = {
            "max_iter": trial.suggest_int("max_iter", 100, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-3, 100.0, log=True
            ),
            "max_bins": trial.suggest_int("max_bins", 32, 255),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        }
        model = make_pipeline(X, classifier_params, args.random_state)
        scores = cross_val_score(
            model,
            X,
            y,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
        )
        trial.set_user_attr("fold_scores", scores.tolist())
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.random_state),
    )
    study.optimize(objective, n_trials=args.n_trials)

    print(f"Rows used for tuning: {len(X):,}")
    print(f"Best mean CV ROC AUC: {study.best_value:.6f}")
    print("Best fold ROC AUC scores:", study.best_trial.user_attrs["fold_scores"])
    print("Best classifier parameters:")
    print(json.dumps(study.best_params, indent=2))


if __name__ == "__main__":
    main()