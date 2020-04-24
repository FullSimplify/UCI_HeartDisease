import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbp', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']


def coverage_ratio(df, plot=False):
    """ calculates the coverage ratio for each column
        in the input dataframe 'df' 

        Args:
            df (pandas.DataFrame)
            plot (bool): if True displays a column plot of the coverage ratios

        Returns:
            dict: dictionary with column names as keys and coverage
                  ratios as values
    """

    cov = {}
    for i, col in enumerate(df.columns):
        cov[col] = len(df[col].isna())/len(df)


    if plot is True:
        f, ax = plt.subplots(figsize=(12, 6))
        ax.bar(cov.keys(), cov.values())
        plt.show()

    return cov




def toNumeric(df):
    """ converts the columns of df to numeric values 
    
        Args:
            df(pandas.DataFrame)
    """
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    


if __name__ == "__main__":

    cleveland = pd.read_csv('processed.cleveland.data', names=COLS) 
    va = pd.read_csv('processed.va.data', names=COLS) 
    hungarian = pd.read_csv('reprocessed.hungarian.data', names=COLS, sep=" ") 
    switzerland = pd.read_csv('processed.switzerland.data', names=COLS) 

    df = pd.concat([cleveland, va, hungarian, switzerland])
    # df.replace("?", np.nan, inplace=True)
    
    toNumeric(df)


    # exclude num != 0, 1
    df = df[(df.num == 1) | (df.num == 0)]
    df['num'] = df.num.astype(int)

    # cov_ratio = coverage_ratio(df, plot=True)

    # plot the distributions
    dist_plots(df)

    # Boxplots
    # box_plots(df)

    
    pass