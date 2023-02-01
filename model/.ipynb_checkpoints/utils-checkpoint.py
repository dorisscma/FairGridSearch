accuracy_metrics = ['acc_score', 'bacc_score', 'f1_score', 'auc_score', 'mcc_score','norm_mcc_score']
group_fairness = ['spd_score', 'aod_score', 'eod_score','ford_score','ppvd_score']
individual_fairness = ['consistency_score','gei_score','ti_score']
fairness_metrics = group_fairness+individual_fairness

bm_category = {'PRE':['RW', 'LFR_pre'],
               'IN': ['LFR_in','AD','EGR'],
               'POST':['ROC','CEO']}
from aif360.sklearn.preprocessing import ReweighingMeta, LearnedFairRepresentations


def store_metrics(y_test, y_pred, X_test, pred_prob, thres_dict, BM_name, threshold, priv_group, pos_label):
    """Returns a dictionary with all interested accuracy and fairness metrics.
        Args:
            y_test (array-like): true labels from test set.
            y_pred (array-like): predicted labels for test set.
            thres_dict (dict): dictionary that stores all info.
            threshold (np.float): given threshold used to obtain y_pred.
        Returns:
            dict: `thres_dict`
    """
    # evaluate model performance for each split
    # --------------------------------------- Accuracy Metrics ---------------------------------------
    thres_dict[BM_name][threshold]['acc_score'] += [accuracy_score(y_test, y_pred)]
    thres_dict[BM_name][threshold]['bacc_score'] += [balanced_accuracy_score(y_test, y_pred)]
    thres_dict[BM_name][threshold]['f1_score'] += [f1_score(y_test, y_pred)]
    thres_dict[BM_name][threshold]['auc_score'] += [roc_auc_score(y_test, pred_prob)]
    thres_dict[BM_name][threshold]['mcc_score'] += [matthews_corrcoef(y_test, y_pred)]
    thres_dict[BM_name][threshold]['norm_mcc_score'] += [0.5*(matthews_corrcoef(y_test, y_pred)+1)]
    # ------------------------------------- Group Fairness Metrics ------------------------------------
    thres_dict[BM_name][threshold]['spd_score'] += [statistical_parity_difference(y_test, y_pred, prot_attr='race',
                                                                                  priv_group=priv_group, pos_label=pos_label)]
    thres_dict[BM_name][threshold]['aod_score'] += [average_odds_difference(y_test, y_pred, prot_attr='race',
                                                                            priv_group=priv_group, pos_label=pos_label)]
    thres_dict[BM_name][threshold]['eod_score'] += [equal_opportunity_difference(y_test, y_pred, prot_attr='race',
                                                                                 priv_group=priv_group, pos_label=pos_label)]
    thres_dict[BM_name][threshold]['ford_score'] += [aif_difference(false_omission_rate_error, y_test, y_pred,
                                                                    prot_attr='race', priv_group=priv_group, pos_label=pos_label)]
    thres_dict[BM_name][threshold]['ppvd_score'] += [aif_difference(precision_score, y_test, y_pred, prot_attr='race',
                                                                    priv_group=priv_group, pos_label=pos_label)]
    # ---------------------------------- Individual Fairness Metrics ----------------------------------
    thres_dict[BM_name][threshold]['consistency_score'] += [consistency_score(X_test, y_pred)]
    thres_dict[BM_name][threshold]['gei_score'] += [generalized_entropy_index(b=y_pred-y_test+1)] # ref: speicher_unified_2018
    thres_dict[BM_name][threshold]['ti_score'] += [theil_index(b=y_pred-y_test+1)]

    return thres_dict

def get_avg_metrics(thres_dict):
    """Returns the average of all cv splits from the same model setting (hyperparameter and threshold).
    Args:
        thres_dict (dict): the dictionary with all info on each cv split.
    Returns:
        dict: `final_metrics`
    """ 
    import copy
    # calculate the average for each metrics from all splits
    avg_metrics = copy.deepcopy(thres_dict)
    for BM in avg_metrics.keys():
        for threshold in avg_metrics[BM].keys(): 
            average_list = {}
            for metric in avg_metrics[BM][threshold].keys():
                average_list['avg_%s'%metric] = mean(avg_metrics[BM][threshold][metric])
            avg_metrics[BM][threshold]['average'] = average_list
    return avg_metrics

def get_output_table(all_metrics, base, scoring):
    """Returns the output table from all param_grid.
    Args:
        all_metrics (dict): the final dictionary with info from all param_grid.
        base (str): the name of the base estimator that is shown in the output table.
    """ 

    output_table = pd.DataFrame()
    for model in all_metrics.keys():
        all_metrics[model]['parameters']['hparam'].pop('random_state', None)
        table_cv = pd.DataFrame(all_metrics[model]['metrics']['average'], index=[0])
        table_cv.insert(0, 'base_estimator', base)
        table_cv.insert(1, 'param', str(all_metrics[model]['parameters']['hparam']))
        table_cv.insert(2, 'Bias_Mitigation', str(all_metrics[model]['parameters']['Bias_Mitigation']))
        table_cv.insert(3, 'threshold', all_metrics[model]['parameters']['threshold'])
        # table_cv[['base_estimator','param']] = np.where(table_cv['Bias_Mitigation'].isin(bm_category['IN'])&\
        #                                                 (table_cv['Bias_Mitigation']!='EGR'), 
        #                                                 '', table_cv[['base_estimator','param']])
        output_table = pd.concat([output_table, table_cv]).reset_index(drop=True)
    
    # find "best" model
    acc_metric = 'avg_'+scoring[0].lower()+'_score'
    fair_metric = 'avg_'+scoring[1].lower()+'_score'
    w_acc = scoring[2]
    w_fair = scoring[3]
    acc_cost = 1-output_table[acc_metric]
    fair_cost = output_table[fair_metric]

    output_table['cost'] = w_acc*acc_cost + w_fair*fair_cost
    return output_table

def style_table(df):
    """Returs the output table with highlight on the best metrics
    Args:
        df (DataFrame): the output table to be styled
    """
    avg_accuracy_metrics = ['avg_'+col for col in accuracy_metrics]
    avg_fairness_metrics = ['avg_'+col for col in fairness_metrics]

    best_index = np.argmin(df.cost)
    df = df.style.highlight_max(subset=avg_accuracy_metrics,color='lightgreen')\
                   .apply(lambda s:['background: yellow' if abs(cell)==min(abs(s)) else '' for cell in s],
                          subset=avg_fairness_metrics)\
                   .highlight_min(subset=['cost'],color='lightblue')\
                   .apply(lambda s:['font-weight: bold' if v == s.iloc[best_index] else '' for v in s])
    
    return df

def merge_dictionary_list(dict_list):
    result_dict = {}
    for d in dict_list:
        for BM in d:
            if BM not in result_dict:
                result_dict[BM] = {}
            # result_dict[BM].append(d[BM])
            for thres in d[BM]:
                # print(thres)
                if thres not in result_dict[BM]:
                    result_dict[BM][thres] = {}
                # result_dict[BM][thres].append(d[BM][thres])
                for metric in d[BM][thres]:
                    # print(thres)
                    if metric not in result_dict[BM][thres]:
                        result_dict[BM][thres][metric] = []
                    result_dict[BM][thres][metric].append(d[BM][thres][metric].pop())

    return result_dict

def LFR(X_train, y_train, X_test, priv_group=1, random_state=1234):
    model = LearnedFairRepresentations(prot_attr='race', random_state=random_state)
    model.fit(X_train, y_train, priv_group=priv_group)
    pred_prob_all = model.predict_proba(X_test)
    
    return pred_prob_all