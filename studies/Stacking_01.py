## This file can be re-run over and over to continue searching.

# CHOOSE THE STUDY
STUDY_NAME = 'Stacking_01'

## CHOOSE MAXIMUM RUNNING TIME:
HOURS = 0
MINUTES = 3
SECONDS = 5

## CHOOSE NUMBER OF TRIALS:
N_TRIALS = 2000

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
train_full = pd.read_csv('../new_datasets/train_{}.csv'.format(STUDY_NAME), index_col=0)
cv_full = pd.read_csv('../new_datasets/cv_{}.csv'.format(STUDY_NAME), index_col=0)


# Load study
study = joblib.load("{}.pkl".format(STUDY_NAME))
total_seconds = pd.read_csv('{}_seconds.csv'.format(STUDY_NAME), index_col=0)

# Load the global_variables
global_variables = pd.read_csv('../global_variables.csv', index_col=0)

SEED = global_variables.loc[0, 'SEED']


# The function to minimize
def train_evaluate(params):
    # Choose variables to include

    features = []
    features_number = 0
    for key, value in params.items():
        if value:
            features.append(key)
            features_number += 1

    # Create the train set
    train = train_full[features]
    # Create the cv set
    cv = cv_full[features]

    # UNCOMMENT TO INSTALL
    # !pip install xgboost
    import xgboost as xgb

    # Instantiate the regressor
    model = xgb.XGBRegressor(random_state=SEED, n_jobs=-1)


    if features_number > 0:
        # Fit
        model.fit(train, train_full['emission'])
        # Calculate the cross-validation Score
        cross_score = mean_squared_error(cv_full['emission'], model.predict(cv), squared=False)

    # If there are no features, then the score is super high
    else:
        cross_score = 1000

    return cross_score


# The function with the parameters ranges. The ranges can be changed.
def objective(trial):
    params = {param: trial.suggest_categorical(param, [True, False]) \
              for param in train_full.drop('emission', axis=1).columns
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
contour_plot = vis.plot_contour(study, params=["pred_1_train", "emission_pred_2_04"])
contour_plot.show()
