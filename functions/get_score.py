def get_score(global_variables, train, test=None, model=None, scores_df=None,
              update=True, comment='',
              prepare_submission=True,
              n_splits=3, global_n_splits=True):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import pandas as pd
    '''
    This function takes the global variables DataFrame, 
    the processed train and test sets,
    an estimator for cross validation,
    a number of cross-validation splits and
    the scores dataframe.
    
    If 'update' is True, then the scores dataframe is being updated with new scores and 'comment'.
    
    If 'prepare_submission' is True, a submission DataFrame is returned
    
    'n_splits' can be chosen for TimeSeriesSplit In this case,'global_n_splits' has to be set to False
    
    It returns:
        1) Average training Score.
        2) Average cross-validation Score
        3) Standard Deviation of cross-validation scores
        4) A submission DataFrame (if 'prepare_submission' is True)
        
    (Score is described in CO2 Emissions.ipynb -> 00. Baseline)
    '''

    # Import global n_splits
    if global_n_splits:
        n_splits = global_variables.loc[0, 'N_SPLITS']

    # Create a TimeSeriesSplit object ('n_splits' splits with equal proportion of positive target values)
    skf = TimeSeriesSplit(n_splits=n_splits)

    # Empty lists for collecting scores
    train_scores = []
    cv_scores = []

    # Iterate through folds
    for train_index, cv_index in skf.split(train.drop('emission', axis=1), train['emission']):
        # Obtain training and testing folds
        cv_train, cv_test = train.iloc[train_index], train.iloc[cv_index]

        # Fit the model
        model.fit(cv_train.drop('emission', axis=1), cv_train['emission'])

        # Calculate scores and append to the scores lists
        train_pred = model.predict(cv_train.drop('emission', axis=1))
        train_scores.append(mean_squared_error(cv_train['emission'], train_pred, squared=False))
        cv_pred = model.predict(cv_test.drop('emission', axis=1))
        cv_scores.append(mean_squared_error(cv_test['emission'], cv_pred, squared=False))

    # Calculate Scores
    train_score = np.mean(train_scores) + np.std(train_scores)
    cross_score = np.mean(cv_scores) + np.std(cv_scores)

    # Update the scores DataFrame
    if update:
        scores_df.loc[len(scores_df)] = [comment, train_score, cross_score, np.nan]

    submission = "prepare_submission=False"

    if prepare_submission:
        # Fit the model to the whole training set:
        model.fit(train.drop('emission', axis=1), train['emission'])
        # Prepare the submission DataFrame
        test_pred = model.predict(test)
        test_pred = pd.DataFrame(test_pred, columns=['emission'])
        submission = pd.concat([pd.DataFrame(test.index, columns=['ID_LAT_LON_YEAR_WEEK']), test_pred], axis=1)

    return train_score, cross_score, np.std(cv_scores), submission