import pandas as pd
from sklearn.metrics import mean_squared_error

def weighted_mean(emission, train_predictions, modificators_dict):

    # Create a DataFrame with predictions Test RMSE's
    prediction_scores = pd.DataFrame({'Score': [32.41069, 26.88799, 21.67107, 30.09157]},
                                    index=train_predictions.columns)

    # Calculate lowest Test RMSE
    min_score = prediction_scores['Score'].min()
    min_score_index = prediction_scores['Score'].idxmin()

    # Calculate weights
    for pred in train_predictions.columns:
        prediction_scores['Weight'] = prediction_scores["Score"] / min_score * modificators_dict[pred]

    # Calculate differences of predictions compared to the best prediction
    train_predictions_differences = train_predictions.subtract(train_predictions[min_score_index], axis=0)

    # Weight differences
    train_predictions_differences_weighted = pd.DataFrame(columns=train_predictions.columns)
    for pred in train_predictions.columns:
        train_predictions_differences_weighted[pred] = train_predictions_differences[pred] / \
                                                    prediction_scores.loc[pred, 'Weight']

    ## Add weighted differences
    train_predictions_weighted = pd.DataFrame(columns=train_predictions.columns)
    for pred in train_predictions.columns:
        train_predictions_weighted[pred] = train_predictions[min_score_index] +\
                        train_predictions_differences_weighted[pred]

    ## Calculate RMSE
    score = mean_squared_error(emission,
                                    train_predictions_weighted.mean(axis=1), squared=False)

    return(score)
