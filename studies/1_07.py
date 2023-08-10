## This file can be re-run over and over to continue searching.

# CHOOSE THE STUDY
STUDY_NAME = '1_07'

## CHOOSE MAXIMUM RUNNING TIME:
HOURS = 0
MINUTES = 15
SECONDS = 5

## CHOOSE NUMBER OF TRIALS:
N_TRIALS = 10000

# CHOOSE THE NUMBER OF PROCESSORS (will be multiplied by 2?)
N_JOBS = 1

# Import packages
import joblib
import pickle
import optuna.visualization as vis
import pandas as pd
import lightgbm as lgb

RUNNING_TIME = HOURS * 3600 + MINUTES * 60 + SECONDS

# Choose the dataset
train = pd.read_csv('../new_datasets/train_{}.csv'.format(STUDY_NAME), index_col=0)

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

    # Calculate the cross-validation Score
    from functions.get_score import get_score

    train_score, cross_score, std, sub = get_score(global_variables, train, model=model, update=False, prepare_submission=False)

    return cross_score


# The function with the parameters ranges. The ranges can be changed.
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 10, 100, step=10),
        'max_depth': trial.suggest_int('max_depth', 2, 30, step=2),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 6e-1, log=True),
        #'n_estimators': trial.suggest_int('n_estimators', 5, 100, step=5),
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
print("Best average cross-validation Score:", study.best_trial.value)
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
contour_plot = vis.plot_contour(study, params=["learning_rate", "min_child_weight"])
contour_plot.show()
