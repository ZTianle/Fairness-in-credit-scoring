# Progress Report

## Preliminary data analysis

We used the latest questionnaire data, downloaded from a public UK data service, that relates to small and medium-sized businesses (SMEs). The raw data includes information on over 45,018 survey results between 2018 and 2020 with almost 153 questions, including the current financial status and their banking relationships. There are some features with the high proportion of missing values, and any technique to impute them will most likely result in inaccurate and biased results.

<!---Initial data exploration reveals the following:-->

### Identify Target Variable

Based on the understanding of the data, our target variable would be a new crafted variable `outcome`, as results are saved as different features in terms of application types.

Additionally, we have the following loan status values

1. You were offered the facility you wanted and took it
2. You took the (PRODUCT) after issues, for example with the terms and conditions or the amount offered
3. You took a different finance product from the (BANK/FINANCE PROVIDER)
4. You were offered finance by (BANK/FINANCE PROVIDER) but decided not to take it
5. You were turned down for finance by (BANK/FINANCE PROVIDER)
6. You are waiting to hear

We keep the data, the status of which belongs to 1-5, and dispose of those with a missing and unknown outcome. Moreover, we classify loans with loan status 5 (turned down) as being in default (or 0), and the other values 1-4 will be classified as good (or 1)，whether or not to take it.

| outcome | amount | proportion |
| :-----: | :----: | :--------: |
|  good   |  2010  |  0.044649  |
| default |  252   |  0.005598  |
| unknown |  167   |  0.003710  |
| missing | 42589  |  0.946044  |

### Data Split

As we can see, we have 2262 data with the final outcome in total. Then we split our data into the following sets: training (80%) and test (20%). We will perform Repeated Stratified k Fold testing on the training test to preliminary evaluate our model while the test set will remain untouched till final model evaluation. This approach follows the best model evaluation practice.

## Data Processing 

### Data cleaning

Next we completed some data cleaning tasks on both training set and test set.

### Feature engineering

Next up, we performed feature selection to identify the most suitable features for our binary classification problem.

Weight of Evidence (WoE) and Information Value (IV) are used for feature engineering and selection and are extensively used in the credit scoring domain.

WoE is a measure of the predictive power of an independent variable in relation to the target variable. It measures the extent a specific feature can differentiate between target classes, in our case: good and bad customers.

IV assists with ranking our features based on their relative importance.

#### Weight of Evidence (WoE)
The formula to calculate WoE is as follow:
$$
W o E_i=\ln \left(\frac{\% \text { of good }}{\% \text { of default }}\right)
$$

Calculate WoE for each unique value (bin) of a categorical variable, e.g., for each of gender: Men, gender: Women, etc.

Once WoE has been calculated for each bin of both categorical and numerical features, combine bins as per the following rules (called coarse classing)

#### Information Value (IV)
IV is calculated as follows:
$$
I V=\sum(\% \text { of good }-\% \text { of default }) \times W o E
$$

By convention, the values of IV in credit scoring is interpreted as follows:
| Information Value      | Variable Predictability |
| :---------: | :---------: |
| less than 0.02  | Not useful for prediction  |
| 0.02 to 0.1   | Weak predictive power        |
| 0.1 to 0.3   | Medium predictive power       |
| 0.3 to 0.5   | Strong predictive power       |
| greater than 0.5   | Suspicious predictive power |



Finally, we updated the predictability of each variable in the variable list.xlsx, and variables with not useful predictability are excluded for next our classification task.

## Model Training and Testing

Finally, we fit a regression model on our training set and evaluate it using k-fold cross validation

Our AUROC on test set comes out to 0.9500 with a Gini of 0.8999 both being considered as quite acceptable evaluation scores.





1.扩大数据集

2.讨论imputation 方法对这个的影响）



