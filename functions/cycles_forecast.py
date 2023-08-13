## CHOOSE IF RUNNING FOR CROSS-VALIDATION, FOR FINAL, OR FOR BOTH
# RUNNING_FOR = 'cv'
# RUNNING_FOR = 'final'
RUNNING_FOR = 'both'

import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import matplotlib.pyplot as plt



global_variables = pd.read_csv('../global_variables.csv', index_col=0)
SEED = global_variables.loc[0, 'SEED']

train_and_cv_unfilled = pd.read_csv('../new_datasets/train_and_cv_unfilled_1_18.csv',
                                    index_col='ID_LAT_LON_YEAR_WEEK')
train_unfilled = pd.read_csv('../new_datasets/train_unfilled_1_18.csv',
                                    index_col='ID_LAT_LON_YEAR_WEEK')
cv_unfilled = pd.read_csv('../new_datasets/cv_unfilled_1_18.csv',
                                    index_col='ID_LAT_LON_YEAR_WEEK')
test_unfilled = pd.read_csv('../new_datasets/test_unfilled_1_18.csv',
                                    index_col='ID_LAT_LON_YEAR_WEEK')
train_cv_test_unfilled = pd.concat([train_and_cv_unfilled, test_unfilled])

y_train_and_cv = pd.read_csv('../new_datasets/y_train_and_cv_1_18.csv',
                           index_col='ID_LAT_LON_YEAR_WEEK')
y_train = pd.read_csv('../new_datasets/y_train_1_18.csv',
                           index_col='ID_LAT_LON_YEAR_WEEK')
y_cv = pd.read_csv('../new_datasets/y_cv_1_18.csv',
                           index_col='ID_LAT_LON_YEAR_WEEK')
y_test = pd.read_csv('../new_datasets/y_test_1_18.csv',
                           index_col='ID_LAT_LON_YEAR_WEEK')
y_train_cv_test = pd.concat([y_train_and_cv, y_test])

# Create a DataFrame with locations and groups


locations_groups = train_unfilled.groupby('Location')['Location_group'].agg(pd.Series.mode)

print(locations_groups.value_counts())

# # Calculate the desired amount of weeks in the cross-validation set
# train_and_cv_weeks = len(train_and_cv_unfilled['WeekCount'].unique())
# cv_weeks = len(cv_unfilled['WeekCount'].unique())
# train_weeks = len(train_unfilled['WeekCount'].unique())
# test_weeks = len(test_unfilled['WeekCount'].unique())
#
# print('Weeks in the training and cv sets: {}'.format(train_and_cv_weeks))
# print('Weeks in the training set: {}'.format(train_weeks))
# print('Weeks in the cross-validation set: {}'.format(cv_weeks))
# print('Weeks in the test set: {}'.format(test_weeks))
#

def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)

model = MultiOutputRegressor(xgb.XGBRegressor(random_state=SEED, n_jobs=-1, n_estimators=100))


def cycles_forecast(y_train_and_test, y_train, y_test,
                    train_and_test_locations, train_locations, test_locations):
    location_count = 0
    # For each location
    for location in train_locations.unique():

        train_and_test_index = train_and_test_locations == location
        train_index = train_locations == location
        test_index = test_locations == location

        ## Set up the lags DataFrame depending on Group

        # Cycles for Special 1 is just 0's
        if locations_groups[location] == 'Special 1':
            y_train.loc[train_index, 'Cycles_forecast'] = 0
            y_test.loc[test_index, 'Cycles_forecast'] = 0

            continue
        elif locations_groups[location] == 'Low':
            emission_1_10_lags_train = \
                pd.DataFrame(y_train.loc[train_index, ['emission_1_10_lag_1', 'emission_1_10_lag_2']])

        else:
            emission_1_10_lags_train = \
                pd.DataFrame(y_train.loc[train_index, 'emission_1_10_lag_1'])

        y = pd.Series(y_train.loc[train_index, 'emission_1_10'])
        # Make multistep target
        y = make_multistep_target(y, steps=sum(test_index)).dropna()

        #print(y)

        # Only keep weeks for which we have both targets and features.
        y, X = y.align(emission_1_10_lags_train, join='inner', axis=0)

        # Fit
        model.fit(X, y)

        ## Create predictions

        # Create prediction matrix
        pred_matrix = pd.DataFrame(model.predict(emission_1_10_lags_train), index=y_train[train_index].index,
                                   columns=y.columns)
        # print(pred_matrix.info())

        # Add empty rows
        empty_matrix = pd.DataFrame(index=y_test[test_index].index, columns=y.columns)
        pred_matrix = pd.concat([pred_matrix, empty_matrix])

        # print(pred_matrix)

        # Shift predictions to align with dates
        pred_matrix_shifted = pd.DataFrame(index=y_train_and_test[train_and_test_index].index)
        for i in range(sum(test_index)):
            pred_matrix_shifted[i + 1] = pred_matrix.iloc[:, i].shift(i + 1)

        #print(pred_matrix_shifted)
        # Calculate mean predictions
        predictions = pred_matrix_shifted.mean(axis=1)
        # Fill missing values
        predictions.fillna(0, inplace=True)

        # Add values to sets
        y_train.loc[train_index, 'Cycles_forecast'] = \
            predictions[:sum(train_index)]
        y_test.loc[test_index, 'Cycles_forecast'] = predictions[sum(train_index):]

        assert (y_train.loc[train_index, 'Cycles_forecast'].isna().any() == False)
        assert (y_test.loc[test_index, 'Cycles_forecast'].isna().any() == False)

        location_count +=1
        print(location_count)

    return y_train, y_test


if RUNNING_FOR == 'cv':
    y_train, y_cv = cycles_forecast(y_train_and_cv, y_train, y_cv,
                                      train_and_cv_unfilled['Location'],
                                      train_unfilled['Location'],
                                      cv_unfilled['Location'])
elif RUNNING_FOR == 'final':
    y_train_and_cv, y_test = cycles_forecast(y_train_cv_test, y_train_and_cv, y_test,
                                      train_cv_test_unfilled['Location'],
                                      train_and_cv_unfilled['Location'],
                                      test_unfilled['Location'])
elif RUNNING_FOR == 'both':
    y_train, y_cv = cycles_forecast(y_train_and_cv, y_train, y_cv,
                                    train_and_cv_unfilled['Location'],
                                    train_unfilled['Location'],
                                    cv_unfilled['Location'])
    y_train_and_cv, y_test = cycles_forecast(y_train_cv_test, y_train_and_cv, y_test,
                                      train_cv_test_unfilled['Location'],
                                      train_and_cv_unfilled['Location'],
                                      test_unfilled['Location'])


y_train_and_cv.to_csv('../new_datasets/y_train_and_cv_1_18_result.csv')
y_train.to_csv('../new_datasets/y_train_1_18_result.csv')
y_cv.to_csv('../new_datasets/y_cv_1_18_result.csv')
y_test.to_csv('../new_datasets/y_test_1_18_result.csv')