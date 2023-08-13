## CHOOSE IF RUNNING FOR CROSS-VALIDATION, FOR FINAL, OR FOR BOTH
RUNNING_FOR = 'cv'
# RUNNING_FOR = 'final'
# RUNNING_FOR = 'both'

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

model = xgb.XGBRegressor(random_state=SEED, n_jobs=-1, n_estimators=100)


def cycles_forecast(y_train_and_test, y_train, y_test,
                    train_and_test_locations, train_locations, test_locations):
    location_count = 0
    # For each location
    for location in train_locations.unique():

        train_and_test_index = train_and_test_locations == location
        train_index = train_locations == location
        train_rows = sum(train_index)
        test_index = test_locations == location


        ## Set up the lags DataFrame depending on Group

        # Cycles for Special 1 is just 0's
        if locations_groups[location] == 'Special 1':
            y_train.loc[train_index, 'Cycles_forecast'] = 0
            y_test.loc[test_index, 'Cycles_forecast'] = 0

            continue


        emission_train = \
            pd.DataFrame(y_train.loc[train_index, ['emission_1_10_lag_1', 'emission_1_10_lag_2', 'emission_1_10']])
        emission_test = \
            pd.DataFrame(index=y_test[test_index].index,
                         columns=['emission_1_10_lag_1', 'emission_1_10_lag_2', 'emission_1_10'])

        emission_df = pd.concat([emission_train, emission_test])

        for index in range(0, sum(test_index)):

            # Calculate lags for the next row
            emission_df.iloc[train_rows - 1 + index + 1, 0] = \
                emission_df.iloc[train_rows - 1 + index, 2]
            emission_df.iloc[train_rows - 1 + index + 1, 1] = \
                emission_df.iloc[train_rows - 1 + index - 1, 2]

            if locations_groups[location] == 'Low':
                # Fit
                model.fit(emission_df.iloc[: train_rows - 1 + index, [0, 1]],
                          emission_df.iloc[: train_rows - 1 + index, 2])
                # Predict next row
                emission_df.iloc[train_rows - 1 + index + 1, 2] = \
                    model.predict(emission_df.iloc[[train_rows - 1 + index + 1], [0, 1]])
            else:
                # Fit
                model.fit(emission_df.iloc[: train_rows - 1 + index, [0]],
                          emission_df.iloc[: train_rows - 1 + index, 2])
                # Predict next row
                emission_df.iloc[train_rows - 1 + index + 1, 2] = \
                    model.predict(emission_df.iloc[[train_rows - 1 + index + 1], [0]])

        # Add values to sets

        y_test.loc[test_index, 'Cycles_forecast'] = emission_df.iloc[train_rows :, 2]

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


y_train_and_cv.to_csv('../new_datasets/y_train_and_cv_1_19_result.csv')
y_train.to_csv('../new_datasets/y_train_1_19_result.csv')
y_cv.to_csv('../new_datasets/y_cv_1_19_result.csv')
y_test.to_csv('../new_datasets/y_test_1_19_result.csv')