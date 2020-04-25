import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import scipy.stats as stats

from utils.data_utils import toNumeric, coverage_ratio
from utils.plot_utils import plot_histograms


COLS = ['age', 'sex', 'cp', 'trestbpd', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']  


if __name__ == "__main__":

    cleveland = pd.read_csv('processed.cleveland.data', names=COLS) 
    va = pd.read_csv('processed.va.data', names=COLS) 
    hungarian = pd.read_csv('reprocessed.hungarian.data', names=COLS, sep=" ") 
    switzerland = pd.read_csv('processed.switzerland.data', names=COLS) 

    df = pd.concat([cleveland, va, hungarian, switzerland])
    
    # df.replace("?", np.nan, inplace=True)
    
    # prepare data
    toNumeric(df)

    # exclude num != 0, 1
    df = df[(df.num == 1) | (df.num == 0)]
    df['num'] = df.num.astype(int)

    # coverage_ratio(df, plot=True)

    # plot the distributions
    # plot_histograms(df)

    # imputation
    # some factors have negative values: 'thal', 'fbs', 'slope', 'oldpeak', 'exang', 'chol'
    # thal must be positive
    # fbs, fasting blood sugar can only be [0, 1] so it is difficult to impute in a meaningful
    # way. Remove the observations with fbs < 0
    # assume that 'thal', 'slope', 'oldpeak' must be non negative

    # fasting blood sugar
    df = df[df.fbs >= 0]

    # we can inpute 'thalach', 'trestbpd', 'ca'
    inputed_vars = ['thal', 'fbs', 'slope', 'oldpeak', 'exang', 'thalach', 'trestbpd', 'ca']
    for var in inputed_vars:
        cond = (df[var] < 0) | (df[var].isna())
        df.loc[cond, var] = df[~cond][var].median()

    # remove also the zero value for cholesterol
    var = 'chol'
    cond = (df[var] <= 0) | (df[var].isna())
    df.loc[cond, var] = df[~cond][var].median()

    # coverage_ratio(df, plot=True)

    # outliers
    # plot the distributions, no visible outliers
    # plot_histograms(df)

    # Split the dataset in training and test    
    labels = df['num'].values
    X = df.drop('num', axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1)

    # XGBoost
    # train without considering the unbalance in the label classes
    # Some of the parameters to look at:
    # booster: gbtree by default
    # eta or learning_rate. At every round the new tree score is multiplied by the learning_rate.
        # A low learning rate makes the model more conservative and less aggressive
    # gamma or min_split_loss (default 0) is the minimum loss reduction require to make a further split.
        # the larger the gamma the less aggressive the model is
    # colsamply_bytree: fraction of features/columns to subsample and use for training.
        # If equal to 1, use all the collumns
    # scale_pos_weight: weight to be used in unbalanced classes dataset.
    params = {
        'booster': 'gbtree',
        'colsample_bynode': .8,
        'learning_rate': .2,
        'max_depth': 2,
        'num_parallel_tree': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'subsample': 0.8,
    } 

    # try with and without weights

    # define the XGBoost input
    # weight. The weights defined in the DMatrix specify the weights assigned to each observation in the training
    # dataset. They are multiplied by the value of the gradient and hessian 
    # (terms in the taylor expansion used to calculate the loss) of the loss function, so the observations
    # with a large weight will "count more" because the gradient update will be more aggressive for that sample.
    # the argument "weight" in DMatrix can be used when the dataset has unbalanced classes.
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=COLS[:-1])
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=COLS[:-1])
    # specify a list of datasets for whihch the the 'eval_metric' will be calculdated during training.
    # if 'evallist' has more than one dataset, the last one is used for early stopping and 
    # I don't want it to be the test set.
    # I want to keep track of the performances on the test set so I put it in the 'evallist'
    # but I don't want to use it for early stopping because if I do it means that I'm using
    # the test set for optimization (through early stopping) and that is a methodological error. 
    evallist = [(dtest, 'test'), (dtrain, 'train')]

    # with the validation set we can specify the 'eraly_stopping_rounds'. The chosen 'eval_metric' is calculated
    # for the last set in 'evallist' and used to determine if the training has to stop. The maximum number of
    # training rounds is 'num_boost_round' but the training is topped early at 'early_stopping_rounds' if the
    # 'eval_metric' does not improve.
    d = {}
    booster = xgb.train(params, dtrain, evals=evallist, evals_result=d, num_boost_round=40, early_stopping_rounds=20)
    
    print("Training with weights")
    # weights = [len(df[df.num == 0])/len(df) if it == 0 else 1 for it in df.num ]
    weights = [0.5 if it == 0 else 1 for it in df.num ]
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=COLS[:-1], weight=weights)
    history2 = xgb.cv(params, dtrain, nfold=5, num_boost_round=40, early_stopping_rounds=20)

    print(history2)

    pass