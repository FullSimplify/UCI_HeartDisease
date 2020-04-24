import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

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
    plot_histograms(df)

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

    coverage_ratio(df, plot=True)

    # outliers
    # plot the distributions
    plot_histograms(df)
    pass