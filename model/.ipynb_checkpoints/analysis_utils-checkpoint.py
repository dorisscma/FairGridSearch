# helper file that collects functions used for analysis for the FairGridSearch results

import pandas as pd

def behaviour_analysis(data, metric_list, category='metric'):
    """
    Analyze how accuracy/fairness changes after applying bias mitigations.
    Args:
        data (dataframe): dataframe that stores all the results 
        metric_list (list): list of metrics to anaylze on, e.g. accuracy/fairness metrics
        category (str): if not "metric", the analysis is run w.r.t. this category, e.g. "BM","base"
    Function that runs the analysis Chen et al. included in their work (fig. 3-6)
    ref: 
    """
    from numpy import mean, std, sqrt
    import scipy.stats as stats

    def cohen_d(x, y):
        return abs(mean(x) - mean(y)) / sqrt((std(x, ddof=1) ** 2 + std(y, ddof=1) ** 2) / 2.0)

    def mann(x, y):
        return stats.mannwhitneyu(x, y)[1]
    
    the_category = category
    # construct diff_degree structure based on the category
    diff_degree = {}
    if the_category == 'metric':
        for metric in metric_list:
            diff_degree[metric] = {}
            for scale in ['noorincrease', 'small', 'medium', 'large']:
                diff_degree[metric][scale] = 0
    elif the_category == 'bm':     
        for BM in data.Bias_Mitigation.unique():
            if BM=='None': pass
            else:
                diff_degree[BM] = {}
                for scale in ['noorincrease', 'small', 'medium', 'large']:
                    diff_degree[BM][scale] = 0
    elif the_category == 'base':
        for base in data.base_estimator.unique():
            diff_degree[base] = {}
            for scale in ['noorincrease', 'small', 'medium', 'large']:
                diff_degree[base][scale] = 0
                
    # analyze changes
    for dataset in data.dataset.unique():
            for metric in metric_list:
                for base in data.base_estimator.unique():
                    default_list = data[(data.dataset==dataset)&\
                                        (data.base_estimator==base)&\
                                        (data.Bias_Mitigation=='None')].reset_index(drop=True)[metric]
                    default_mean = default_list.mean()
                    for BM in data[data.base_estimator==base].Bias_Mitigation.unique():
                        if BM == 'None': pass
                        else: 
                            bm_list = data[(data.dataset==dataset)&\
                                           (data.base_estimator==base)&\
                                           (data.Bias_Mitigation==BM)].reset_index(drop=True)[metric]
                            bm_mean = bm_list.mean()
                            rise_ratio = bm_mean - default_mean
                            cohenn = cohen_d(default_list, bm_list)
                            # store values according to category                    
                            category = metric if the_category == 'metric' else (BM if the_category == 'bm' else base)
                            if mann(default_list, bm_list) >= 0.05 or rise_ratio >= 0:
                                diff_degree[category]['noorincrease'] += 1
                            elif cohenn < 0.5:
                                diff_degree[category]['small'] += 1
                            elif cohenn >= 0.5 and cohenn < 0.8:
                                diff_degree[category]['medium'] += 1
                            elif cohenn >= 0.8:
                                diff_degree[category]['large'] += 1
    table = pd.DataFrame(diff_degree)
    table = table.apply(lambda x: x/x.sum())
    table.columns = [col.removeprefix('abs_').removeprefix('avg_').removesuffix('_score').upper() for col in table.columns]
    # shorten "1-consistency" name
    table = table.rename({'(1-CONSISTENCY_SCORE)': '1-CNS'}, axis=1)
    # re-order columns
    if 'LR' in table.columns:
        table = table[['LR','RF','GB','SVM','NB','TABTRANS']]
    elif 'RW' in table.columns:
        table = table[['RW','LFR_PRE','LFR_IN','AD','EGR','ROC','CEO','RW+ROC','RW+CEO']]

    return diff_degree, table


def plot_behaviour_analysis(table, caption='', figsize=(8, 6)):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns
    sns.set()
    import re
    # %matplotlib inline

    font_color = '#525252'
    # csfont = {'fontname':'Georgia'} # title font
    # hfont = {'fontname':'Calibri'} # main font
    # colors = ['#f47e7a', '#b71f5c', '#621237', '#dbbaa7']
    colors = ['#F8F8F8','#DCDCDC','#A9A9A9','#696969']

    # 1. Create the plot
    data = table.copy().T
    ax = data.plot.bar(align='center', stacked=True, figsize=figsize, color=colors, width=0.3, edgecolor="none")
    plt.tight_layout()

    # 2. Create the title        
    title = plt.title(caption, pad=60, fontsize=18, color=font_color)
    title.set_position([.5, 1.02])
    # Adjust the subplot so that the title would fit
    plt.subplots_adjust(top=0.8, left=0.26)

    # 3. Set labels’ and ticks’ font size and color
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(10)
    plt.xticks(rotation=0, color=font_color)
    plt.yticks(color=font_color)

    # 4. Add legend    
    legend = plt.legend(loc='center',
           frameon=False,
           bbox_to_anchor=(0., 1.02, 1., .102), 
           mode='expand', 
           ncol=4, 
           borderaxespad=-.46,
           prop={'size': 15})

    for text in legend.get_texts():
        plt.setp(text, color=font_color) # legend font color
    # 5. Add annotations      
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        if height>0:
            ax.text(x+width/2, 
                    y+height/2, 
                    '{0:.2f}%'.format(height*100), 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    color='black',
                    # weight='bold',
                    fontsize=8)