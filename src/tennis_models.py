"""Train and score models with a walk-forward test, so every prediction is out of sample."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tennis_features import both_points_of_view

INTERACTIONS = {
    "d_elo_x_best_of_5": ("d_elo_combined", "best_of_5"),
    "d_elo_x_slam": ("d_elo_combined", "level_slam"),
    "d_elo_x_clay": ("d_elo_combined", "is_clay"),
    "d_elo_x_grass": ("d_elo_combined", "is_grass"),
    "d_serve_x_court_aces": ("d_serve_slow", "court_ace_rate"),
    "d_height_x_grass": ("d_height", "is_grass"),
    "d_rest_x_best_of_5": ("d_log_rest", "best_of_5"),
}


def logistic_model(C=0.1):
    # A missing difference becomes 0, meaning "no known advantage to either player"
    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000),
    )


def lasso_model(C):
    """Logistic regression with a LASSO penalty. Smaller C means a stronger penalty."""
    # Each match appears from both points of view, so the intercept is zero by design
    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        StandardScaler(),
        LogisticRegression(penalty="l1", solver="liblinear", C=C, fit_intercept=False, max_iter=500, random_state=0),
    )


def boosting_model(learning_rate=0.03, max_iter=400, max_leaf_nodes=15, min_samples_leaf=200, l2=1.0):
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2, random_state=0,
    )


def add_interactions(features):
    """Add product features for logistic regression, which can't find interactions by itself."""
    result = features.copy()
    for name, (direction, context) in INTERACTIONS.items():
        result[name] = result[direction].fillna(0) * result[context].fillna(0)
    return result


def fit_both_views(model, X):
    """Train on each match from both players' points of view, so the winner isn't always first."""
    both = pd.concat([X, both_points_of_view(X)], ignore_index=True)
    target = np.r_[np.ones(len(X)), np.zeros(len(X))]
    return model.fit(both, target)


def predict_winner_probability(model, X):
    """Probability given to the actual winner, averaged across both points of view."""
    from_winner = model.predict_proba(X)[:, 1]
    from_loser = model.predict_proba(both_points_of_view(X))[:, 1]
    return (from_winner + 1 - from_loser) / 2


def walk_forward(data, columns, make_model, first_train_year=2005, years=range(2015, 2022)):
    """For each year: train on all earlier years, then predict that year."""
    predictions = pd.Series(np.nan, index=data.index, dtype=float)
    for year in years:
        train = data[data["year"].between(first_train_year, year - 1)]
        current = data[data["year"] == year]
        model = fit_both_views(make_model(), train[columns])
        predictions.loc[current.index] = predict_winner_probability(model, current[columns])
    return predictions


def score(winner_probability):
    """Accuracy and log loss from the probability given to each match's actual winner."""
    p = np.clip(np.asarray(winner_probability, dtype=float), 1e-6, 1 - 1e-6)
    accuracy = np.where(p == 0.5, 0.5, p > 0.5).mean()
    return {"accuracy_pct": round(100 * float(accuracy), 2), "log_loss": round(float(-np.log(p).mean()), 4)}


def recalibrate_walk_forward(data, predictions, years=range(2016, 2022)):
    """Correct probabilities using a one-input logistic fit on earlier years' predictions only."""
    logit = np.log(predictions / (1 - predictions))
    corrected = pd.Series(np.nan, index=data.index, dtype=float)
    for year in years:
        earlier = data["year"].between(2015, year - 1) & predictions.notna()
        current = data["year"] == year
        x = logit[earlier].to_numpy().reshape(-1, 1)
        calibrator = LogisticRegression(fit_intercept=False, C=1e6)
        calibrator.fit(np.vstack([x, -x]), np.r_[np.ones(len(x)), np.zeros(len(x))])
        corrected[current] = calibrator.predict_proba(logit[current].to_numpy().reshape(-1, 1))[:, 1]
    return corrected
