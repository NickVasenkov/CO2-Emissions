def get_score_3(global_variables, train, test=None,
              post_training_df=None,
              model=None, scores_df=None,
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
    the post-training DataFrame,
    an estimator for cross validation,
    the scores DataFrame
    and a number of cross-validation splits.
    
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

    TARGET = 'emission_03'

    # Import global n_splits
    if global_n_splits:
        n_splits = global_variables.loc[0, 'N_SPLITS']

    # Create a TimeSeriesSplit object ('n_splits' splits with equal proportion of positive target values)
    skf = TimeSeriesSplit(n_splits=n_splits)

    # Empty lists for collecting scores
    train_scores = []
    cv_scores = []

    # Iterate through folds
    for train_index, cv_index in skf.split(train.drop(TARGET, axis=1), train[TARGET]):
        # Obtain training and testing folds
        cv_train, cv_test = train.iloc[train_index], train.iloc[cv_index]
        post_training_train, post_training_cv = post_training_df.iloc[train_index], \
                                                  post_training_df.iloc[cv_index]

        # Fit the model
        model.fit(cv_train.drop(TARGET, axis=1), cv_train[TARGET])

        # Predictions of residues
        train_pred = model.predict(cv_train.drop(TARGET, axis=1))
        cv_pred = model.predict(cv_test.drop(TARGET, axis=1))

        # transform into the real target
        train_pred += post_training_train['Trend'] + post_training_train['Seasonality']
        cv_pred += post_training_cv['Trend'] + post_training_cv['Seasonality']


        # Calculate scores and append to the scores lists
        train_scores.append(mean_squared_error(post_training_train['emission'], train_pred, squared=False))
        cv_scores.append(mean_squared_error(post_training_cv['emission'], cv_pred, squared=False))

    # Calculate Scores
    train_score = np.mean(train_scores) + np.std(train_scores)
    cross_score = np.mean(cv_scores) + np.std(cv_scores)

    # Update the scores DataFrame
    if update:
        scores_df.loc[len(scores_df)] = [comment, train_score, cross_score, np.nan]

    submission = "prepare_submission=False"

    if prepare_submission:
        post_training_test = post_training_df.loc[test.index]

        # Fit the model to the whole training set:
        model.fit(train.drop(TARGET, axis=1), train[TARGET])

        # Predictions of residues
        test_pred = model.predict(test)


        # transform into the real target
        test_pred += post_training_test['Trend'] + post_training_test['Seasonality']

        # Prepare the submission DataFrame
        test_pred = pd.DataFrame(test_pred, columns=['emission'])

        submission = pd.concat([pd.DataFrame(test.index, columns=['ID_LAT_LON_YEAR_WEEK'], index=test.index), test_pred], axis=1)

    return train_score, cross_score, np.std(cv_scores), submission