## This file can be re-run over and over to continue searching.

# CHOOSE THE STUDY
STUDY_NAME = '2_04'

## CHOOSE MAXIMUM RUNNING TIME:
HOURS = 0
MINUTES = 5
SECONDS = 0

## CHOOSE NUMBER OF TRIALS:
N_TRIALS = 10000

# CHOOSE THE NUMBER OF PROCESSORS (will be multiplied by 2)
N_JOBS = -1

# Import packages
import joblib
import pickle
import optuna.visualization as vis
import pandas as pd
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# CHOOSE THE DATASET
train = pd.read_csv('../new_datasets/train_2_03.csv', index_col='ID_LAT_LON_YEAR_WEEK')
cv = pd.read_csv('../new_datasets/cv_2_03.csv', index_col='ID_LAT_LON_YEAR_WEEK')

global_variables = pd.read_csv('../functions/global_variables.csv', index_col=0)
SEED = global_variables.loc[0, 'SEED']
train_from_part_1 = pd.read_csv('../new_datasets/train_from_part_1.csv', index_col='ID_LAT_LON_YEAR_WEEK')
test_from_part_1 = pd.read_csv('../new_datasets/test_from_part_1.csv', index_col='ID_LAT_LON_YEAR_WEEK')
train['date'] = pd.to_datetime(train_from_part_1['date'])
cv['date'] = pd.to_datetime(test_from_part_1['date'])
top_three_values = train_from_part_1.loc[:, 'Location_enc'].drop_duplicates().sort_values(ascending = False).head(3)
top_three_locations = train_from_part_1.loc[train_from_part_1['Location_enc'].isin(top_three_values), 'Location'].drop_duplicates()
cv_weeks = 37

RUNNING_TIME = HOURS * 3600 + MINUTES * 60 + SECONDS

# Load study
study = joblib.load("{}.pkl".format(STUDY_NAME))
total_seconds = pd.read_csv('{}_seconds.csv'.format(STUDY_NAME), index_col=0)

# A function to minimize
def train_evaluate(params):
    # For each location
    for location in train['Location'].unique():
        # Create time series
        series = pd.Series(train.loc[train['Location'] == location, 'emission_03'].values,
                           index=train.loc[train['Location'] == location, 'date'])
        series.index = series.index.to_period('W')
        # Save indices
        old_train_index = train[train['Location'] == location].index
        old_cv_index = cv[cv['Location'] == location].index

        # sin/cos pairs for annual seasonality
        pairs = params['2']
        if location == top_three_locations[0]:
            pairs = params['0']
        elif location == top_three_locations[1]:
            pairs = params['1']
        fourier = CalendarFourier(freq="A", order=pairs)

        # Set up DeterministicProcess
        dp = DeterministicProcess(index=series.index, constant=True,
                                  order=1,  # trend (order 1 means linear)
                                  seasonal=False,  # indicators
                                  additional_terms=[fourier],  # annual seasonality
                                  drop=True,  # drop terms to avoid collinearity
                                  )

        ## Calculate seasonality on the training set

        # create features for dates in index
        X = dp.in_sample()

        lr = LinearRegression(fit_intercept=False)
        lr.fit(X, series)
        y_pred = pd.Series(lr.predict(X), index=old_train_index)
        train.loc[old_train_index, 'Seasonality_2'] = y_pred

        # Predict seasonality for the cv set
        X = dp.out_of_sample(steps=cv_weeks)
        y_pred = pd.Series(lr.predict(X), index=old_cv_index)
        cv.loc[old_cv_index, 'Seasonality_2'] = y_pred

    # Add back subtractions:
    pred_cv = cv['Seasonality_2'] + cv['Seasonality'] + cv['Trend']

    # Calculate Score
    score_cv = mean_squared_error(cv['emission'], pred_cv, squared=False)

    return score_cv

# The function with the parameters ranges. The ranges can be changed.
def objective(trial):
    params = {
        '0': trial.suggest_int('0', 1, 26),
        '1': trial.suggest_int('1', 1, 26),
        '2': trial.suggest_int('2', 1, 26),

    }
    return train_evaluate(params)


# Run the session
study.optimize(objective, timeout=RUNNING_TIME, n_trials=N_TRIALS, n_jobs=N_JOBS)
total_seconds.iloc[0, 0] = total_seconds.iloc[0, 0] + RUNNING_TIME

# Save back to the file
joblib.dump(study, "{}.pkl".format(STUDY_NAME))
with open('{}_params.pkl'.format(STUDY_NAME), 'wb') as f:
    pickle.dump(study.best_params, f)
total_seconds.to_csv('{}_seconds.csv'.format(STUDY_NAME))


print("Best trial:", study.best_trial.number)
print("Best Score:", study.best_trial.value)
print("Best hyperparameters:", study.best_params)
total_hours = round(total_seconds.iloc[0, 0] / 3600, 3)
print("Total running time (hours):", total_hours)



# Plotting optimization history (INTERACTIVE)
optimization_history_plot = vis.plot_optimization_history(study, error_bar=True)
optimization_history_plot.show()

# Plotting parameter importance
param_importance_plot = vis.plot_param_importances(study)
param_importance_plot.show()

# Plotting a contour plot
contour_plot = vis.plot_contour(study, params=["0", "1"])
contour_plot.show()
