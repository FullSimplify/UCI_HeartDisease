from utils import pd
from utils import MinMaxScaler
from utils import plt
from utils import np


def box_plots(df, metadata):
    """ displays boxplots for each column in the input dataframe 

        Args:
            df (pandas.DataFrame): input dataframe
    """

    scaler = MinMaxScaler()

    f, ax = plt.subplots(1, 1, figsize=(12, 8))    
    k = 1
    names = []
    for i, col in enumerate(df.columns):
        if col != "num":
            names.append("  " + col)
            names.append("")
            c = df[col][(df[col]>0) & (df['num'] == 1)].values.reshape(-1, 1)
            c = scaler.fit_transform(c)
            p1 = ax.boxplot(c[~np.isnan(c)], showfliers=False, positions=[k - .4],
                            patch_artist=True, 
                            boxprops=dict(facecolor='red', color='red', alpha=.5),
                            whiskerprops=dict(color='red'),
                            medianprops=dict(color='black'),
                            widths=(.6))
            # ax.set_yticklabels("")
            
            c = df[col][(df[col]>0) & (df['num'] == 0)].values.reshape(-1, 1)
            c = scaler.fit_transform(c)
            p2 = ax.boxplot(c[~np.isnan(c)], showfliers=False, positions=[k + .4],
                            patch_artist=True,
                            boxprops=dict(facecolor='green', color='green', alpha=.5),
                            whiskerprops=dict(color='green'),
                            medianprops=dict(color='black'),
                            widths=(.6))
            ticks = list(ax.get_xticks())
            ticks[-1] = np.nan
            ax.set_xticks(ticks)
            # ax.set_yticklabels(["", col])
            k += 3


    ax.set_xticklabels(names)
    ax.tick_params(width=0, length=0)
    plt.show(block=False)


def plot_histograms(df):
    """ plots the histograms of the columns of the input dataframe

        Args:
            df(pandas.DataFrame): input data
    """
    cols = df.columns
    ncols = len(cols)
    if ncols%15 == 0:
        nfigs = int(ncols/15)
    else:
        nfigs = int(np.floor(ncols/15) + 1)
        rem = int(ncols%15)
    t = 0 
    for n in range(nfigs):        
        if n < nfigs: 
            f, axes = plt.subplots(5, 3, figsize=(13, 10))
            f.suptitle("Histograms. Red = diseased, green = healthy", fontsize=16)

            k = 0
            for j in range(3):
                for i in range(5):
                    c = cols[i+k+t - 1]
                    axes[i, j].hist(df[c][df["num"] == 0], bins=100, color="green", alpha=.5)
                    axes[i, j].hist(df[c][df["num"] == 1], bins=100, color="red", alpha=.5)
                    axes[i, j].set_xlabel(c)
                k = k + 5                                   
        else:            
            f, axes = plt.subplots(rem, 1, figsize=(6, 10))
            f.suptitle("Histograms. Red = diseased, green = healthy", fontsize=16)
            for i in range(rem):
                c = cols[-rem:][i]
                axes[i].hist(df[c], bins=100)
                axes[i].set_xlabel(c)

        t = t + 10      
        plt.subplots_adjust(hspace=.5, right=.95, left=.05)   

    # plt.tight_layout()
    plt.show()