# CO2 Emissions
 A Kaggle competition (spatial and time data, regression). The competition took place in August 2023.

My solution incorporates a wide range of techniques, including spatial and time feature engineering, Mean Target Encoding, Hyperparameters tuning with Bayesian Search, Clustering, multi-layered deseasoning, etc.

The competition webpage: https://www.kaggle.com/competitions/playground-series-s3e20/overview/description .

 ## Task description

The objective of this challenge is to create machine learning models that use open-source emissions data (from Sentinel-5P satellite observations) to predict carbon emissions.

Approximately 497 unique locations were selected from multiple areas in Rwanda, with a distribution around farm lands, cities and power plants. The data for this competition is split by time; the years 2019 - 2021 are included in the training data, and your task is to predict the CO2 emissions data for 2022 through November.

Note, the data is not real, it was *syntesized based on real data*, that is why we are predicting 2022 emissions in a 2023 competition.

## Results

During the competition, we could see the 'Public Score', which was calculated on a hidden test set. We could make up to 105 submissions during the competition, and the best result is counted towards the 'Public Score' position. By the 'Public Score', I finished in top 11%. (https://www.kaggle.com/competitions/playground-series-s3e20/leaderboard?tab=public , as nvasenkov).

The leaderboard position is calculated at the end of competition, based on a separate hidden test set ('Private Score'). For the Private Score, we need to select only 2 submissions from those we've made during the competition. I finished in top 23%: https://www.kaggle.com/nvasenkov . However, one of the submissions I've done during the competition (but didn't selected it as a final submission)) scored better than the 1st place submission.

## My solution:

My solution is spreaded across 5 Jupiter notebooks, plus several separate Python code files for reusable functions and computationally expensive pieces of code.

The 4 notebooks named CO2 Emissions 1 through 4 represent 4 distinct approaches to the problem:

https://github.com/NickVasenkov/CO2-Emissions/blob/main/CO2%20Emissions%201.ipynb

https://github.com/NickVasenkov/CO2-Emissions/blob/main/CO2%20Emissions%202.ipynb

https://github.com/NickVasenkov/CO2-Emissions/blob/main/CO2%20Emissions%203.ipynb

https://github.com/NickVasenkov/CO2-Emissions/blob/main/CO2%20Emissions%204.ipynb

The notebook 'CO2 Emissions Stacking' represents several techniques of combining the best models from the previous parts into a new model:

https://github.com/NickVasenkov/CO2-Emissions/blob/main/CO2%20Emissions%20Stacking.ipynb

My winning solution by the 'Private Score' is in the 'CO2 Emissions Stacking' notebook, in the paragraph named '07. Weighted Mean'. It is a mean of the forecast of my 4 best models, weighted based on their 'Public Scores'.

My winning solution by the 'Public Score' is in the 'CO2 Emissions 3' notebook, in the paragraph named '05. Max with coeffs for locations'. This approach is using the fact that there are strong annual patterns in CO2 Emissions. For each of 497 locations, I've found (through a Bayesian Search) a multiplier to apply to 2019-2020 emissions in order to make the most accurate forecast of 2021 emissions (available data). Then I applied the found multipliers to 2019-2021 emissions to predict the 2022 emissions (unavailable data). This model received a good "Private Score' by itself, but it scored better in combination with 2 other models (as described in the previous paragraph).

