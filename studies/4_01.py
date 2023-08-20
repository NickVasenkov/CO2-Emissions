## This file can be re-run over and over to continue searching.

# CHOOSE THE STUDY
STUDY_NAME = '4_01'

## CHOOSE MAXIMUM RUNNING TIME:
HOURS = 0
MINUTES = 20
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
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

RUNNING_TIME = HOURS * 3600 + MINUTES * 60 + SECONDS

# Load the datasets
train = pd.read_csv('../new_datasets/train_{}.csv'.format(STUDY_NAME), index_col=0)
tuning = pd.read_csv('../new_datasets/tuning_{}.csv'.format(STUDY_NAME), index_col=0)
y_train = pd.read_csv('../new_datasets/y_train_{}.csv'.format(STUDY_NAME), index_col=0)
y_tuning = pd.read_csv('../new_datasets/y_tuning_{}.csv'.format(STUDY_NAME), index_col=0)


# Load study
study = joblib.load("{}.pkl".format(STUDY_NAME))
total_seconds = pd.read_csv('{}_seconds.csv'.format(STUDY_NAME), index_col=0)

# Load the global_variables
global_variables = pd.read_csv('../global_variables.csv', index_col=0)

SEED = global_variables.loc[0, 'SEED']

# The function to minimize
def train_evaluate(params):

    # Instantiate the classifier
    model = lgb.LGBMRegressor(random_state=SEED, n_jobs=N_JOBS, **params)

    # Fit
    model.fit(train, y_train['emission'])
    # Calculate the tuning Score
    pred_tuning = model.predict(tuning)
    tuning_score = mean_squared_error(y_tuning['emission'], pred_tuning, squared=False)

    return tuning_score


# The function with the parameters ranges. The ranges can be changed.
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 10, 100, step=10),
        'max_depth': trial.suggest_int('max_depth', 2, 30, step=2),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 6e-1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 10, 300, step=10),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 1e3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100, step=10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.00, step=0.05),
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 50, step=10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.00, step=0.05),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1e2, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1e2, log=True)
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
print("Best average tuning Score:", study.best_trial.value)
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
contour_plot = vis.plot_contour(study, params=["num_leaves", "max_depth"])
contour_plot.show()
