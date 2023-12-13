# Power-Outage-Prediction-Project

#### Identify the Problem: How can we predict whether or not a power outage was caused by severe weather?

From a general perspective, the first part to finding a solution (both preventative and responsive) is identifying the cause of the problem. With that in mind, we wanted to find a way to identify whether or not a major power outage was caused by severe weather from data that we'd have at the time of the outage in hopes of restoring the outage faster with that information.

This is a classification problem as we are just trying to classify the cause of a major outage. This is binary classification, as we're just looking to predict whether or not the cause of the outage was severe weather. Our response variable will thus be the cause of the outage, or CAUSE.CATEGORY (once modified to be 0 or 1 depending on whether the cause was severe weather), as our goal is to predict this variable. As noted earlier, we think this information would be helpful in the restoration of outages when the time comes. The metric we are using to evaluate our model will be the accuracy. From looking at the dataset, the proportion of outages caused by severe weather is very close to 0.5, meaning that the accuracy would not fail to capture the full picture of our model's effectiveness due to an imbalance in number of classifications. We'll also be looking at precision and recall to see whether we're producing more false positives or false negatives (or, essentially over or under-predicting the cause as severe weather). One important point to note is that since we would not have information on the following columns at the time of the outage: OUTAGE.RESTORATION, OUTAGE.DURATION, DEMAND.LOSS.MW, CAUSE.CATEGORY.DETAIL, we cannot use this for our prediction. However, we will be incorporating other features such as day and month of the outage as we would know that information at the time of a major power outage.

### Baseline Model

Our first model will be a decision tree classifier. We believe this is a good baseline because it'll serve as a proof of concept for a random forest classifier, while still being effective on its own. We're using a relatively smaller training set (test set makes up 0.3 of the full set), as in our initial tests with the model, we saw it had a tendency to overfit on the training data.

# Features:

1. `YEAR` -- We believe that one of the first things we should look at when building our model is the time at which the outage occured. We think that weather should have a general trend over time, so we're choosing to use this as a numerical variable. Though it shouldn't matter in a decision tree classifier, we thought it was good practice to encode a ***standard scaler*** to our year column.
2. `CLIMATE.REGION` -- We also think that place is an important feature to use when looking at outages. Certain regions may or may not be subject to more harsh weather conditions, which leads us to believe certain regions would have tendencies to have more outages caused by severe weather. We ***one-hot encoded*** this column such that categories could be represented numerically for our model.

As mentioned above, we think trends should occur over time, meaning we're leaving the years numeric. This makes `YEAR` a quantitative variable. The climate region is a nominal variable, as it's a categorical variable with no inherent order.
Therefore, we have ***one*** quantitative variable, ***one*** nominal variable, and ***zero*** ordinal variables in our base model.

Our encodings were done using a ColumnTransformer such that we would be able to apply different encoding methods to the different columns. We also used the `remainder="drop"` argument in our preprocessor, meaning that all other columns were omitted from our model.

# Analysis:

Model Statistics:
1. Accuracy: 0.73
2. Precision: 0.69
3. Recall: 0.72

We consider this model to be passable, but it definitely can be improved upon. The accuracy is 0.73, which is significantly better than that of a naive model which would always predict an outage was not caused by severe weather (and have an accuracy of around 0.51). However, being correct 73% of the time with no extreme tendency towards false positives or negatives (as shown by our precision and recall) would not make the model particularly useful on a practical basis (i.e. in the event of a major power outage). Therefore, we want to look into other features that can better our model.

### Final Model

Our final model will be a random forest classifier. We believe that the tree voting method will improve the steps in the decision tree, leading to a stronger model overall.

# Added Features

We left the original features from the base model in our model. However, we did have some adds to attempt to increase the accuracy:

1. `CLIMATE.CATEGORY` -- We added climate category because this represents the climate episodes by year in each region. This means it provides context as to the climate relative to both time and location at the same time -- a good indicator for our model. Because climate is the "long-term pattern of weather" (via National Geographic), we assumed that certain climates would lead to more severe weather leading to outages. This is a categorical, nominal variable (no inherent order), so we ***one-hot encoded*** this column.
2. `ANOMALY.LEVEL` -- This variable is similar to `CLIMATE.CATEGORY`, as it represents the climate episdes relative to both time and location -- this time, by season. This would provide a bit more granularity for understanding the climate, which we ultimately decided would improve the model. Because this variable is quantitative, we applied a ***standard scaler*** to this column.
3. `CUSTOMERS.AFFECTED` -- We wanted to first note that due to the existence of NaN values, we had to impute those. We chose to impute those using the median, as the mean was skewed due to the fact that there was a signficant number of entries below 75,000, but a few far far above. Imputing by the median is an attempt to "dampen" the effect of the NaN values. We think, though, that the scale of the outage could be indicative of its cause. For example, an outage that wipes out a large number of people's electricity is more likely to be due to severe weather because it requires large infrastructural damage. Because this variable is quantitative, we applied a ***standard scaler*** to this column.
4. `TOTAL.CUSTOMERS` -- We think that total customers in the given state of the outage could be indirectly related with the cause of the outage. We think this because a greater demand for electricity could lead to stronger infrastructure being implemented, leading to more resilience particularly against severe weather. This was also a quantitative variable, so we applied a ***standard scaler*** to this column.

# Methods: Testing Models

1. ***Decision Tree Classifier*** -- We tried this model with the expectation that it would be slightly worse than the random forest classifier. Its accuracy was 0.87.
2. ***SVC*** -- We tried this model knowing that it tends to be less effective than the decision tree or the random forest classifier. Its accuracy was 0.78.
3. ***Random Forest Classifier*** -- As noted above, a random forest classifier is often a stronger model than the decision tree. We found this in the performance, leading us to select that.

# Tuning Hyperparameters

1. ***Max Depth*** -- We think that max depth is an important hyperparameter to tune because we noticed that in our base model, our training accuracy was much higher than our testing accuracy -- indicating that overfitting was an issue. Restricting the max depth can help with overfitting because having less steps and thresholds could mean that our model is less sensitive to outliers or variation in the dataset. Therefore, we fitted our max depth.
2. ***n_estimators*** -- As mentioned above, we noticed some overfitting with our model. We also know that the number of trees voting on the final model can affect the fitness of the set, as a greater number of trees could lead to sensitivity to rather insignificant variation in our data.

To tune hyperparameters, we began by just messing around with numbers in different ranges to see approximately what range works best both for the max depth and the number of estimators. We ultimately found that the best max depth was low (< 30), as the models tended to be overfitted. We found that n estimators was best north of 500, which makes sense considering that a greater number of trees voting would result in a better overall model.

We then went on to use a GridSearchCV to try and fit the model, resulting in an accuracy of 0.87 with a depth of 24 and a number of estimators of 200. These hyperparameters didn't make sense in terms of what we had tested. We figured that the smaller training sets, though covering all possibilities and averaging, led to different accuracies. Therefore, we then implemented a loop that manually tested hyperparameters -- depth between 2 and 30, and number of estimators between 100 and 1000. This resulted in a greater accuracy, as we'll discuss below.

Our final parameters:
1. ***Max Depth***: 10
2. ***n_estimators***: 900

# Analysis

Model Statistics:
1. Accuracy: 0.8937
2. Precision: 0.90
3. Recall: 0.84

We consider this model to be significantly more useful than our base model, with this model being 22% more accurate than our base (relative to the base rate). Being able to predict with 89% accuracy could give people relative confidence as to the cause of the outage and allow them to react accordingly. Though it's not 100%, we do think that on occassion, severe weather outages and intentional attacks for example could have similar effects in similar times/locations, leading to this 11% gap. However, we were impressed with the improvement over our base model.

#### Accuracy Analysis

To do our fairness assessment, we will categorize our dataset into two groups: major power outages that affected less than 75,000 people and major power outages that affected more than 75,000 people. For this analysis we will classify power outages affecting more than 75,000 as a sever power outage. For our analysis our primary metric will be accuracy Our proposed null hypothesis is that our model's accuracy for determining a severe outage is roughly equivalent to an outage that is not severe. Our proposed alternative hypothesis is that our model is unfair, in that it has a higher accuracy for outages that are not severe than those that are classified as severe based on the amount of people affected. The test statistic we have chosen will be the difference in accuracy between severe and non-sever power outages, with a significance level of 0.01. For this analysis, we ran a permutation test 1,000 times and obtained a p-value of 0.001, which is within our significance level. This p-value leads us to reject the null hypothesis, indicating that our alternative hypothesis that the model has a higher accuracy for power outages that affect less than 75,000 people than power outages that affect more than 75,000 power outages. However we cannot assert with certaintly that our model is unfair as the permutation tests are created through random choice. Because of this we advise to conduct further testing with more data and permutations to determine whether or not the model is "truly fair".
