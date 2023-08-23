## This file can be re-run over and over to continue searching.

# CHOOSE THE STUDY
STUDY_NAME = '3_05'

## CHOOSE MAXIMUM RUNNING TIME:
HOURS = 2
MINUTES = 30
SECONDS = 5

## CHOOSE NUMBER OF TRIALS:
N_TRIALS = 10000

# CHOOSE THE NUMBER OF PROCESSORS (will be multiplied by 2?)
N_JOBS = -1

# Import packages
import joblib
import pickle
import optuna.visualization as vis
import pandas as pd
from sklearn.metrics import mean_squared_error

RUNNING_TIME = HOURS * 3600 + MINUTES * 60 + SECONDS

# Choose the dataset
train = pd.read_csv('../new_datasets/train_3_02.csv'.format(STUDY_NAME), index_col=0)

# Load study
study = joblib.load("{}.pkl".format(STUDY_NAME))
total_seconds = pd.read_csv('{}_seconds.csv'.format(STUDY_NAME), index_col=0)

# Load the global_variables
global_variables = pd.read_csv('../global_variables.csv', index_col=0)

SEED = global_variables.loc[0, 'SEED']


# The function to minimize
def train_evaluate(params):

    # Calculate RMSE
    emission = train.loc[train['year'] == 2021, 'emission']

    # Multiply 'Max_train' by coeffs
    train.loc[train['year'] == 2021, 'Max_multiplied_train'] = \
        train.loc[train['year'] == 2021, 'Location'].map(params) * \
        train.loc[train['year'] == 2021, 'Max_train']
    # print(len(train.loc[train['year'] == 2021, 'Max_multiplied_train'].unique()))

    score = mean_squared_error(emission, train.loc[train['year'] == 2021, 'Max_multiplied_train'], squared=False)

    return score


# The function with the parameters ranges. The ranges can be changed.
def objective(trial):
    params = {location: trial.suggest_float(location, 0.9, 1.1, step=0.001) for location in \
                                            train['Location'].unique()}
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
print("Best average cross-validation Score:", study.best_trial.value)
print("Best hyperparameters:", study.best_params)
total_hours = round(total_seconds.iloc[0, 0] / 3600, 3)
print("Total running time (hours):", total_hours)

# Plotting optimization history (INTERACTIVE)
optimization_history_plot = vis.plot_optimization_history(study, error_bar=True)
optimization_history_plot.show()

# # Plotting parameter importance
# param_importance_plot = vis.plot_param_importances(study)
# param_importance_plot.show()

# # Plotting a contour plot
# contour_plot = vis.plot_contour(study, params=["High", "Low"])
# contour_plot.show()
