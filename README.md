# Tennis Match Prediction

Predicting ATP tennis match outcomes using only pre-match information, and testing whether the model can find value against bookmaker odds.

**Stack:** Python, pandas, SQL (SQLite), scikit-learn

## Project structure
```
data/raw/        original CSV files (downloaded, not in Git)
data/processed/  SQLite database and cleaned tables
notebooks/       step-by-step analysis, one notebook per phase
src/             Elo, feature, model and market code
app/             the Streamlit app
reference/       tournament countries, written by hand
```

## Run the app
Pick two players to see the model's win probabilities and fair odds, then enter any bookmaker's odds to compare them. Another tab keeps a paper-trading log, with no real money.

```bash
pip install -r requirements.txt
```

```bash
python app/build_snapshot.py
```

```bash
streamlit run app/app.py
```

The first time, run notebooks 01 and 05 to download the match data, and `python src/download_tml.py` to download the newer matches. The app only knows matches up to the last data update, and it says so when that data is more than 14 days old.

A Calendar tab shows the 2026 season's progress. To refresh the model and the calendar:

```bash
python src/download_tml.py
```

```bash
python app/build_snapshot.py
```

```bash
python app/build_calendar.py
```

The Practice bets tab only appears when the app runs on your own computer.

## Data
Match data: created by Jeff Sackmann (`tennis_atp`), via the archive mirror [tennis-sackmann-archive](https://github.com/Aneeshers/tennis-sackmann-archive) (CC BY-NC-SA 4.0).
Includes main tour, qualifying, Challenger and Futures matches.
Newer 2026 matches: [TennisMyLife](https://stats.tennismylife.org/tennis-match-database), for personal, educational and non-commercial research use. Its files can't be redistributed, so they aren't in this repository. The Calendar tab uses only a summary of dates and winners, in `app/calendar_events.csv`, and the model file `app/app_snapshot.joblib` holds ratings and statistics calculated from these matches.
Betting odds: [tennis-data.co.uk](http://www.tennis-data.co.uk).

This is a non-commercial educational project. The match data isn't in this repository, and the values in `app/app_snapshot.joblib` calculated from Sackmann's data stay under CC BY-NC-SA 4.0.

### Extending the data with TennisMyLife
`python src/update_data.py` adds the matches after Sackmann's last ones to a separate database, `data/processed/tennis_latest.db`. Among the notebooks, only notebook 12 uses it. The model in the app loads the same matches directly, so it's retrained on them and hasn't been tested on newer matches yet.

- **Added:** 1,352 main-tour matches from June to 5 November 2026, plus Challenger and qualifying matches for the ratings.
- **Past unchanged:** all 78,876 earlier matches keep exactly the same values.
- **Quality:** on matches in both sources, about 99% of scores agree.
- **Limits:** there are no Futures matches after June 2026. About 1.7% of player slots in main-tour matches are linked by name and age, a rule that was wrong for 3 of 802 players with a known id, and the dates of matches are estimated as they are for older years.

## Results
Final test on 12,215 unseen main-tour matches, from 2022 to June 2026:

| Method | Accuracy | Log loss |
|---|---|---|
| ATP ranking only | 63.81% | 0.6310 |
| Elo only | 65.87% | 0.6083 |
| **Final model** | **66.98%** | **0.5929** |

Against Pinnacle's odds on the same 10,375 matches (2022–2025), the model scores 66.57% and 0.5959, while Pinnacle scores 67.71% and 0.5889. A value-betting rule fixed in advance returned -1.22% (95% interval: -5.28% to +3.08%), so **the model shows no proven edge over the market**.

## Further analysis
Notebook 09 uses matches up to 2021 only, so the final test stays untouched:

- **LASSO:** no better at predicting (log loss 0.5849 for both models), but it shows that Elo replaces the ATP ranking.
- **Event study of retirements:** compared with the same player's ordinary losses, win probability is 5.49 points lower in the next match (95% interval: -6.71 to -4.22). This isn't proof of a causal effect of injury, and it affects too few matches to improve predictions much.

Notebook 10 looks for errors in the bookmakers' prices:

- **One bias repeats:** of 63 variables, only seeding adds to Pinnacle's price. Better-seeded players win slightly less than the price suggests, on new years and on the women's tour.
- **No proven profit:** a rule fixed in advance returned +4.40% for men (2022-2025) and -0.52% for women, and both intervals include zero.
- **A lead to test:** at the best available price the rule looks better, but this is exploratory, and those prices may not have been available to bet.

Notebook 11 tests NVIDIA's pretrained Kumo Tabular model, using a rule fixed in advance:

- **Effectively a tie:** on 6,460 main-tour matches from 2019 to 2021, Kumo's log loss is 0.5951 and our model's is 0.5942. The difference is −0.00083 (95% interval: −0.00342 to +0.00186), so Kumo doesn't pass.
- **Limited setting:** Kumo saw only 4,000 past matches because of laptop memory, while our model's columns were chosen with these years as validation. The result applies to this setting, not to Kumo in general.
- **Decision:** we keep our model, which is simpler and easier to explain. Notebook 11 reads saved predictions, which `src/kumo_test.py` creates in the separate environment from `requirements-kumo.txt`.

Notebook 12 is a one-time test on 1,253 main-tour matches from June to November 2026, which the model had never seen, using rules fixed in advance:

- **Lower score:** 65.28% accuracy and a log loss of 0.6243. Elo also fell, to 63.77% and 0.6343, so this period looks harder.
- **Elo, only just:** the model's gain is +0.0100, and the lower end of its 95% interval is +0.00001.
- **Market, no:** on 1,202 matches, the market average scores 0.6153 and the model 0.6221 (gain −0.0068, 95% interval: −0.0165 to +0.0027). Pinnacle has no prices in this period.
- **Betting, not shown:** the rule at Bet365 returned +0.81% on 426 bets (95% interval: −11.66% to +13.91%).

## Progress
- [x] Phase 0 – Setup
- [x] Phase 1 – Understand the data
- [x] Phase 2 – SQL database
- [x] Phase 3 – Exploration with SQL
- [x] Phase 4 – Feature engineering (Elo, form, head-to-head)
- [x] Phase 5 – Modelling
- [x] Phase 6 – Evaluation vs bookmakers & backtest
- [ ] Phase 7 – Presentation
