
from utils import plt
from utils import pd


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
        cov[col] = sum(df[col].notna())/len(df)


    if plot is True:
        f, ax = plt.subplots(figsize=(12, 6))
        ax.bar(cov.keys(), cov.values())
        plt.show(block=True)

    return cov




def toNumeric(df):
    """ converts the columns of df to numeric values 
    
        Args:
            df(pandas.DataFrame)
    """
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')