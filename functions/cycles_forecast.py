import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import matplotlib.pyplot as plt

global_variables = pd.read_csv('../global_variables.csv', index_col=0)
SEED = global_variables.loc[0, 'SEED']
train_and_cv = pd.read_csv('../new_datasets/train_and_cv_02_04.csv', index_col='ID_LAT_LON_YEAR_WEEK')
test = pd.read_csv('../new_datasets/test_02_04.csv', index_col='ID_LAT_LON_YEAR_WEEK')
train_cv_test = pd.concat([train_and_cv, test])

train_and_cv_weeks = len(train_and_cv['WeekCount'].unique())
test_weeks = len(test['WeekCount'].unique())


def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)

model = MultiOutputRegressor(xgb.XGBRegressor(random_state=SEED, n_jobs=-1, n_estimators=70))


location_count=0
# For each location
for location in train_and_cv['Location'].unique():
    # Save indices
    train_cv_test_index = train_cv_test[train_cv_test['Location'] == location].index
    train_and_cv_index = train_and_cv[train_and_cv['Location'] == location].index
    test_index = test[test['Location'] == location].index

    # DataFrames

    emission_03_lags_train_and_cv = \
        pd.DataFrame(train_and_cv.loc[train_and_cv_index, ['emission_03_lag_1', 'emission_03_lag_2']])
    y = pd.Series(train_and_cv.loc[train_and_cv_index, 'emission_03'])

    # Make multistep target
    y = make_multistep_target(y, steps=test_weeks).dropna()

    # Only keep weeks for which we have both targets and features.
    y, X = y.align(emission_03_lags_train_and_cv, join='inner', axis=0)

    # Fit
    model.fit(X, y)

    ## Create predictions

    # Create prediction matrix
    pred_matrix = pd.DataFrame(model.predict(emission_03_lags_train_and_cv), index=train_and_cv_index,
                               columns=y.columns)
    # Add empty rows
    empty_matrix = pd.DataFrame(index=test_index, columns=y.columns)
    pred_matrix = pd.concat([pred_matrix, empty_matrix])
    # Shift predictions to align with dates
    pred_matrix_shifted = pd.DataFrame(index=train_cv_test_index)
    for i in range(test_weeks):
        pred_matrix_shifted[i + 1] = pred_matrix.iloc[:, i].shift(i + 1)
    # Calculate mean predictions
    predictions = pred_matrix_shifted.mean(axis=1)
    # Fill missing values
    predictions.fillna(0, inplace=True)

    # Add values to sets
    train_and_cv.loc[train_and_cv_index, 'Cycles_forecast'] = \
        predictions[:train_and_cv_weeks]
    test.loc[test_index, 'Cycles_forecast'] = predictions[train_and_cv_weeks:]

    assert(train_and_cv.loc[train_and_cv_index, 'Cycles_forecast'].isna().any() == False)
    assert(test.loc[test_index, 'Cycles_forecast'].isna().any() == False)

    print(location_count)
    location_count += 1

    # predictions.plot()
    # train_and_cv.loc[old_train_and_cv_index, 'emission_03_lag_1'].plot()
    # train_and_cv.loc[old_train_and_cv_index, 'emission_03'].plot()
    # plt.show()


train_and_cv.to_csv('../new_datasets/train_and_cv_02_04_result.csv')
test.to_csv('../new_datasets/test_02_04_result.csv')