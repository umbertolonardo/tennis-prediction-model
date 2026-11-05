"""Test NVIDIA's Kumo Tabular against our model. Notebook 11 reads the predictions this saves.

Needs its own environment (see requirements-kumo.txt). Run from the project folder:
    python src/kumo_test.py --tune    chooses the setting on 2018 only
    python src/kumo_test.py           runs the fixed test on 2019-2021, which can take hours on a laptop
"""
import argparse
import gc
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sdm  # noqa: E402
from tennis_domain_features import DOMAIN_GROUPS  # noqa: E402
from tennis_features import FEATURE_GROUPS, both_points_of_view  # noqa: E402
from tennis_models import INTERACTIONS, add_interactions, logistic_model, score, walk_forward  # noqa: E402

ORDER = ["Elo", "Ranking", "Serve and return", "Form", "Fatigue and fitness", "Experience",
         "Style and age", "Head to head", "Draw", "Home"]
COLUMNS = ([c for group in ORDER for c in FEATURE_GROUPS[group]]
           + list(INTERACTIONS) + DOMAIN_GROUPS["Tournament history"])
TEST_YEARS = [2019, 2020, 2021]
BATCH = 100  # Matches predicted at a time, to limit memory use


def load_matches():
    """Main-tour matches up to 2021. Each row is written from the winner's point of view."""
    conn = sqlite3.connect(ROOT / "data/processed/tennis.db")
    data = pd.read_sql_query("SELECT * FROM final_features WHERE tourney_level <> 'D' AND year <= 2021", conn)
    assert data["year"].is_monotonic_increasing, "rows must be in date order for the context to be the latest matches"
    return add_interactions(data).reset_index(drop=True)


def win_column(frame):
    """Find the column holding the probability of class 1, and refuse anything unexpected."""
    names = [c for c in frame.columns if str(c).split("_")[-1] == "1"]
    assert len(names) == 1 and frame.shape[1] == 2, list(frame.columns)
    return names[0]


def predict_batch(model, context, query, device, estimators):
    """Kumo's probability for each match's actual winner, averaged across both points of view."""
    x_context = pd.concat([context[COLUMNS], both_points_of_view(context[COLUMNS])], ignore_index=True).astype(float)
    x_context["target"] = np.r_[np.ones(len(context), dtype=int), np.zeros(len(context), dtype=int)]
    x_query = pd.concat([query[COLUMNS], both_points_of_view(query[COLUMNS])], ignore_index=True).astype(float)
    x_query["target"] = 0  # Placeholder so both tables share one schema. It's dropped before predicting.
    table = pd.concat([x_context, x_query], ignore_index=True)
    stypes = sdm.infer_stypes(table, overrides={"target": "categorical"})
    tensor = sdm.TableTensor.from_pandas(df=table, stypes=stypes, device=device)
    n = len(x_context)
    with torch.no_grad():
        probs = model(x_context=tensor[:n].drop_columns("target"), y_context=tensor[:n, "target"],
                      x_query=tensor[n:].drop_columns("target"), num_estimators=estimators)
    frame = probs.to_pandas()
    win = frame[win_column(frame)].to_numpy(dtype=float)
    half = len(query)
    return (win[:half] + 1 - win[half:]) / 2


def predict_year(model, data, year, context_size, device, estimators):
    """Predict one year from the latest matches before it, so the model never sees the future."""
    context = data[data["year"] < year].tail(context_size)
    query = data[data["year"] == year]
    results = []
    for start in range(0, len(query), BATCH):
        results.append(predict_batch(model, context, query.iloc[start:start + BATCH], device, estimators))
        gc.collect()
        print(f"{year}: {min(start + BATCH, len(query))}/{len(query)}", flush=True)
    return pd.Series(np.concatenate(results), index=query.index)


def tune(data, device):
    """Compare three settings on 2018, a year outside the test."""
    ours = walk_forward(data, COLUMNS, logistic_model, years=[2018])[data["year"] == 2018]
    print("Our model, 2018:", score(ours))
    for size, context_size in [("small", 4000), ("large", 2000), ("large", 4000)]:
        model = sdm.models.KumoTabular(task="classification", size=size, device=device)
        print(f"Kumo {size}, {context_size} matches, 2018:", score(predict_year(model, data, 2018, context_size, device, 1)))


def test(data, device):
    """Run the setting fixed before 2019-2021 were opened, and save the predictions."""
    ours = walk_forward(data, COLUMNS, logistic_model, years=TEST_YEARS)
    model = sdm.models.KumoTabular(task="classification", size="small", device=device)
    kumo = pd.concat([predict_year(model, data, year, 4000, device, 8) for year in TEST_YEARS])
    table = data.loc[kumo.index, ["tourney_id", "match_num", "year"]].assign(ours=ours[kumo.index], kumo=kumo)
    table.to_csv(ROOT / "data/processed/kumo_predictions.csv", index=False)
    print("Saved", len(table), "matches. Ours:", score(table["ours"]), "Kumo:", score(table["kumo"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="choose the setting on 2018 instead of running the test")
    # The Mac's GPU ran out of memory as batches accumulated, so the CPU is the default
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    (tune if args.tune else test)(load_matches(), torch.device(args.device))
