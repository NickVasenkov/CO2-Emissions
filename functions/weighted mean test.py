import pandas as pd

predictions = pd.DataFrame({'pred_1': [-1, 1, -1],
                            'pred_2': [2, 2, 2],
                            'pred_3': [3, -3, 3]})

print(predictions)

prediction_scores = pd.DataFrame({'Score': [10, 8, 9]},
                                index=predictions.columns)

min_score = prediction_scores['Score'].min()

print(min_score)

min_score_index = prediction_scores['Score'].idxmin()

#print(min_score_index)
#print(predictions[min_score_index])


prediction_scores['Weight'] = prediction_scores["Score"] / min_score

#print(prediction_scores)

predictions_differences = predictions.subtract(predictions[min_score_index], axis=0)

print(predictions_differences)

predictions_differences_weighted = pd.DataFrame(columns=predictions.columns)

for pred in predictions.columns:
    predictions_differences_weighted[pred] = predictions_differences[pred] / prediction_scores.loc[pred, 'Weight']

print(predictions_differences_weighted)

predictions_weighted = pd.DataFrame(columns=predictions.columns)

for pred in predictions.columns:
    predictions_weighted[pred] = predictions[min_score_index] + predictions_differences_weighted[pred]

print(predictions_weighted)
