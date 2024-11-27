import numpy as np

def list_of_dicts_to_dataframe(lst_results, st_key):
    # expects lists of results, returns a long-form dataframe of one 
    # extra result keys are retained in the dataframe
    import pandas as pd

    dfs = []

    for result in lst_results:
        df = pd.DataFrame(result[st_key])
        
        df.columns = ['value']
        if 'Condensed Spectrum' in st_key:
            df['degree'] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.5, 13.5, 17.5, 23.0, 29.5, 38.0, 49.0, 63.0, 81.0, 104.5, 134.5, 173.0, 223.0]
        elif 'Spectrum' in st_key:
            df['degree'] = np.arange(252)
        for k, val in result.items():
            if not isinstance(val, np.ndarray):
                df[k] = val
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df



def plot_condensed_spectra(df, groupkey=None, unitkey=None, palette=None):
    """ Plots a seaborn plot of a long-form pandas dataframe by groupkey and unitkey and palette
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.text import Text
    wavelength = 1/np.sqrt(df['degree'] * (df['degree']-1)) 
    df['wavelength'] = wavelength
    df.loc[df['degree'] == 1,'wavelength'] = 3 # to make inf plottable
    df.loc[df['degree'] == 2,'wavelength'] = 1 # fix coarse wavelength
    
    f, ax = plt.subplots(1,1,figsize=(4.6, 2.8))
    plt.subplots_adjust(wspace=0, hspace=0) 
    sns.lineplot()

    # Beautify going to infinite
    df_inf = df[df['wavelength'] >=1]
    df['to inf'] = False
    df_inf['to inf'] = True
    df = df[df['wavelength'] != 3]
    df = pd.concat([df,df_inf], ignore_index=True)

    if groupkey is not None:
        if unitkey is not None:
            sns.lineplot(data=df, ax = ax, x="wavelength", y='value',units=unitkey, estimator=None, hue=groupkey, palette=palette, legend=None, alpha=1, linewidth=0.4, style='to inf',  dashes=[(4,2),(2,2)])  
        sns.lineplot(df, ax = ax, x="wavelength", y='value', hue=groupkey, palette=palette, legend='full', errorbar=None, linewidth=1.5, style='to inf', dashes=[(1,0),(2,1)])  
    else:
        if unitkey is not None:
            sns.lineplot(data=df, ax = ax, x="wavelength", y='value',units=unitkey, estimator=None, hue=unitkey, palette=palette, alpha=1, linewidth=1, style='to inf',  dashes=[(1,0),(2,1)]) 
        else:   
            sns.lineplot(df, ax = ax, x="wavelength", y='value', palette=palette, legend='full', errorbar=None, linewidth=1.5, style='to inf', dashes=[(1,0),(2,1)])  
    sns.despine()

    plt.ylim(1e-5, 0.1)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xticks([1.e-03, 1.e-02, 1.e-01, 1, 3, 100])
    ax.set_xticklabels([Text(0.001, 0, '$\\mathdefault{10^{-3}}$'), "0.01", "0.1", "1", "inf", Text(100.0, 0, '$\\mathdefault{10^{2}}$')])
    ax.set_xlim(0.003,4)
    ax.set_xlabel("Approximate wavelength (rad/2π)")
    ax.set_ylabel("Contribution to variance")
    ax.set_ylim(1e-6, 1)

    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.show()

