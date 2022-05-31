from cmath import nan
from unicodedata import category
import scipy, pandas, numpy
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_selection import f_classif
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso,  ElasticNetCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, RocCurveDisplay
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

category_notuseful = ['q11a', 'q15z', 'q85']

# custom transformer class to create new categorical dummy features
class WoE_Binning(BaseEstimator, TransformerMixin):
    def __init__(self, X): # no *args or *kargs
        self.X = X
    def fit(self, X, y = None):
        return self #nothing else to do
    def transform(self, X):
        return X
    def preprocess(self, X):
        # 1-white, 2-mixed, 3-asian, 4-black, missing
        X['q144_new']=nan
        X.loc[(X['q144']==1)|(X['q144']==2)|(X['q144']==3),'q144_new'] = 1
        X.loc[(X['q144']==4)|(X['q144']==5)|(X['q144']==6)|(X['q144']==7),'q144_new'] = 2
        X.loc[(X['q144']==8)|(X['q144']==9)|(X['q144']==10)|(X['q144']==11)|(X['q144']==15),'q144_new'] = 3
        X.loc[(X['q144']==12)|(X['q144']==13)|(X['q144']==14),'q144_new'] = 4
        X.drop(['q144'], axis=1, inplace=True)

        #those variables without missing values 
        X = pandas.get_dummies(X, columns=['risk'])

        X = pandas.get_dummies(X, columns=['q126'])

        # X = pandas.get_dummies(X, columns=['q144'])

        X = pandas.get_dummies(X, columns=['q7q8'])

        X = pandas.get_dummies(X, columns=['q9'])

        X = pandas.get_dummies(X, columns=['q11'])

        X = pandas.get_dummies(X, columns=['q12'])

        X = pandas.get_dummies(X, columns=['q13'])

        X = pandas.get_dummies(X, columns=['q14y'])

        X = pandas.get_dummies(X, columns=['q14ysu2'])

        X = pandas.get_dummies(X, columns=['q15d2'])

        X = pandas.get_dummies(X, columns=['q15z'])

        X = pandas.get_dummies(X, columns=['q24a'])

        X = pandas.get_dummies(X, columns=['q24b'])

        X = pandas.get_dummies(X, columns=['q24c'])

        X = pandas.get_dummies(X, columns=['q78'])

        X = pandas.get_dummies(X, columns=['q103106'])

        X = pandas.get_dummies(X, columns=['q120'])


        #those variables with missing values but handled 
        X = X.drop(X[X['q15_1'] == -99.99].index)
        X = pandas.get_dummies(X, columns=X.filter(like='q15_').columns.tolist())

        df_temp  = X.copy()
        cols = df_temp.filter(like='q35b').columns.tolist()+df_temp.filter(like='q53').columns.tolist()
        X[cols] = df_temp[cols].replace(-99.99,0.00,inplace=False)

        #those variables with missing values but not handled 
        X = pandas.get_dummies(X, columns=['q11a'])

        X = pandas.get_dummies(X, columns=['q13a'])

        X = pandas.get_dummies(X, columns=['q13b'])

        X = pandas.get_dummies(X, columns=['q14a'])

        X = pandas.get_dummies(X, columns=X.filter(like='q15b_').columns.tolist())

        X = pandas.get_dummies(X, columns=['q15c'])

        X = pandas.get_dummies(X, columns=X.filter(like='q17_').columns.tolist())

        X = pandas.get_dummies(X, columns=X.filter(like='q26_').columns.tolist())

        X = pandas.get_dummies(X, columns=['q27'])

        X = pandas.get_dummies(X, columns=X.filter(like='q28_').columns.tolist())

        df_temp  = X.copy()
        cols = df_temp.filter(like='q38').columns.tolist()+df_temp.filter(like='q56').columns.tolist()
        X[cols] = df_temp[cols].replace([-99.99, 3.00],0.00,inplace=False)

        X['q38_56'] = -99.99
        for i in cols:
            X.loc[X[i] == 1.00,'q38_56'] = 1.00
            X.loc[X[i] == 2.00,'q38_56'] = 2.00
            X.drop(i,axis=1,inplace=True)

        X = pandas.get_dummies(X, columns=['q38_56'])

        df_temp  = X.copy()
        cols = df_temp.filter(like='q42').columns.tolist()+df_temp.filter(like='q60').columns.tolist()
        X[cols] = df_temp[cols].replace([-99.99, 3.00],0.00,inplace=False)

        X['q42_60'] = -99.99
        for i in cols:
            X.loc[X[i] == 1.00,'q42_60'] = 1.00
            X.loc[X[i] == 2.00,'q42_60'] = 2.00
            X.drop(i,axis=1,inplace=True)

        X = pandas.get_dummies(X, columns=['q42_60'])

        df_temp  = X.copy()
        cols = df_temp.filter(like='q43').columns.tolist()+df_temp.filter(like='q61').columns.tolist()
        X[cols] = df_temp[cols].replace([-99.99, 3.00],0.00,inplace=False)

        X['q43_61'] = -99.99
        for i in cols:
            X.loc[X[i] == 1.00,'q43_61'] = 1.00
            X.loc[X[i] == 2.00,'q43_61'] = 2.00
            X.drop(i,axis=1,inplace=True)

        X = pandas.get_dummies(X, columns=['q43_61'])

        X = pandas.get_dummies(X, columns=['q78b'])

        X = pandas.get_dummies(X, columns=X.filter(like='q78c').columns.tolist())

        X = pandas.get_dummies(X, columns=X.filter(like='q81').columns.tolist())

        X = pandas.get_dummies(X, columns=X.filter(like='q84_').columns.tolist())

        X = pandas.get_dummies(X, columns=['q85'])

        X = pandas.get_dummies(X, columns=['q111112'])

        X = pandas.get_dummies(X, columns=['q113'])

        X = pandas.get_dummies(X, columns=X.filter(like='qbb').columns.tolist())

        X['q115_116']  = -99.99
        X.loc[X['q115'] == 3.00,'q115_116'] = 0.00
        X.loc[X['q116_p'] == 1.00,'q115_116'] = 1.00
        X.loc[X['q116_p'] == 2.00,'q115_116'] = 2.00
        X.loc[X['q116_p'] == 3.00,'q115_116'] = 3.00
        X.loc[X['q116_p'] == 4.00,'q115_116'] = 4.00
        X.loc[X['q116_p'] == 5.00,'q115_116'] = 5.00
        X.loc[X['q116_l'] == 1.00,'q115_116'] = -1.00
        X.loc[X['q116_l'] == 2.00,'q115_116'] = -2.00
        X.loc[X['q116_l'] == 3.00,'q115_116'] = -3.00
        X.loc[X['q116_l'] == 4.00,'q115_116'] = -4.00
        X.loc[X['q116_l'] == 5.00,'q115_116'] = -5.00

        X = pandas.get_dummies(X, columns=['q115_116'])

        X = pandas.get_dummies(X, columns=['q117'])

        X = pandas.get_dummies(X, columns=['q119'])


        X.drop('Unnamed: 0', axis=1, inplace=True)    
        for i in category_notuseful:
            cols = X.filter(like=i).columns.tolist()

            for j in cols:
                X.drop(j,axis=1,inplace=True)           

        return X

if __name__=="__main__":
    df = pandas.read_csv('sme_finance_monitor_preprocess.csv')
    # split data into 80/20 while keeping the distribution of bad loans in test set same as that in the pre-split dataset

    woe_transform = WoE_Binning(df)

    X = woe_transform.preprocess(df)
    
    y = X['outcome']

    X = X.drop(['outcome'], axis=1)
    

    print(X['q144_new'].value_counts())

     # KNN imputer
    # imputer = SimpleImputer(strategy="most_frequent")
    imputer = IterativeImputer(IterativeImputer(estimator=RandomForestRegressor(random_state=0), max_iter=20))
    # imputer = KNNImputer(n_neighbors=5, weights="uniform")
    X_new = imputer.fit_transform(X, y)
    print(X['q144_new'])
    print(X_new[:,-1])
    # X = pandas.get_dummies(X, columns=['q144_new'])

    # X = SimpleImputer(strategy="most_frequent")

    # mask = numpy.random.randint(0, 2, size=X['126'].shape).astype(bool)
    # print(mask)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, 
                                                        random_state = 42, stratify = y)
 
     # hard copy the X datasets to avoid Pandas' SetttingWithCopyWarning when we play around with this data later on.
    # this is currently an open issue between Pandas and Scikit-Learn teams
    X_train, X_test = X_train.copy(), X_test.copy()

    # define modeling pipeline
    # reg = LogisticRegression(max_iter=100000)
    # reg = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, class_weight='balanced')
    # reg = LinearRegression()
    reg = Lasso(alpha=0.05)
    # reg = clf = MLPClassifier(activation='logistic', solver='adam', alpha=1e-5, hidden_layer_sizes=(500,), random_state=1)

    # reg =  ElasticNetCV(cv=5, random_state=0)

    pipeline = Pipeline(steps=[('woe', woe_transform), ('model', reg)])

    # define cross-validation criteria
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

    # fit and evaluate the regression pipeline with cross-validation as defined in cv
    scores = cross_val_score(pipeline, X_train, y_train, scoring = 'roc_auc', cv = cv)
    
    
    AUROC = numpy.mean(scores)
    
    GINI = AUROC * 2 - 1

    # print the mean AUROC score and Gini
    print('Mean AUROC: %.4f' % (AUROC))
    print('Gini: %.4f' % (GINI))

    # fit the pipeline on the whole training set
    pipeline.fit(X_train, y_train)

    # create a summary table
    # first create a transformed training set through our WoE_Binning custom class
    X_train_woe_transformed = woe_transform.fit_transform(X_train)
    # Store the column names in X_train as a list
    # feature_name = X_train_woe_transformed.columns.values

    # Create a summary table of our logistic regression model
    # summary_table = pandas.DataFrame(columns = ['Feature name'], data = feature_name)
    # # Create a new column in the dataframe, called 'Coefficients'
    # summary_table['Coefficients'] = numpy.transpose(pipeline['model'].coef_)
    # # Increase the index of every row of the dataframe with 1 to store our model intercept in 1st row
    # summary_table.index = summary_table.index + 1
    # # Assign our model intercept to this new row
    # summary_table.loc[0] = ['Intercept', pipeline['model'].intercept_[0]]
    # Sort the dataframe by index
    # summary_table.sort_index(inplace = True)

    y_pred = pipeline.predict(X_test)
    RocCurveDisplay.from_predictions(y_test, y_pred)
    plt.show()

    # # make preditions on our test set
    # y_hat_test = pipeline.predict(X_test)
    # # get the predicted probabilities
    # y_hat_test_proba = pipeline.predict_proba(X_test)
    # # select the probabilities of only the positive class (class 1 - default) 
    # y_hat_test_proba = y_hat_test_proba[:][: , 1]

    # # we will now create a new DF with actual classes and the predicted probabilities
    # # create a temp y_test DF to reset its index to allow proper concaternation with y_hat_test_proba
    # y_test_temp = y_test.copy()
    # y_test_temp.reset_index(drop = True, inplace = True)
    # y_test_proba = pandas.concat([y_test_temp, pandas.DataFrame(y_hat_test_proba)], axis = 1)
    # # Rename the columns
    # y_test_proba.columns = ['y_test_class_actual', 'y_hat_test_proba']
    # # Makes the index of one dataframe equal to the index of another dataframe.
    # y_test_proba.index = X_test.index

    # # get the values required to plot a ROC curve
    # fpr, tpr, thresholds = roc_curve(y_test_proba['y_test_class_actual'], 
    #                                 y_test_proba['y_hat_test_proba'])
    # # plot the ROC curve
    # plt.plot(fpr, tpr)
    # # plot a secondary diagonal line, with dashed line style and black color to represent a no-skill classifier
    # plt.plot(fpr, fpr, linestyle = '--', color = 'k')
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve');

    # # Calculate the Area Under the Receiver Operating Characteristic Curve (AUROC) on our test set
    # AUROC = roc_auc_score(y_test_proba['y_test_class_actual'], y_test_proba['y_hat_test_proba'])
    # # calculate Gini from AUROC
    # Gini = AUROC * 2 - 1
    # # print AUROC and Gini
    # print('AUROC: %.4f' % (AUROC))
    # print('Gini: %.4f' % (Gini))

    # # draw a PR curve
    # # calculate the no skill line as the proportion of the positive class
    # no_skill = len(y_test[y_test == 1]) / len(y)
    # # plot the no skill precision-recall curve
    # plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # # get the values required to plot a PR curve
    # precision, recall, thresholds = precision_recall_curve(y_test_proba['y_test_class_actual'], 
    #                                                     y_test_proba['y_hat_test_proba'])
    # # plot PR curve
    # plt.plot(recall, precision, marker='.', label='Logistic')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend()
    # plt.title('PR curve');