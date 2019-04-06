import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from scipy import stats


def data_summary(input_df):
    nrows, ncols = input_df.shape
    df_summary = input_df.describe(percentiles = [0.1, 0.25, 0.5, 0.75, 0.90], 
                                   include='all').T
    df_summary.drop(['top', 'freq'], axis=1, inplace=True)
    df_summary['data_type'] = input_df.dtypes
    df_summary['null_percentage'] = ((nrows - df_summary['count']) * 100) / nrows
    for i in list(input_df.columns):
        df_summary.loc[i, 'value_counts'] = str(input_df[i].value_counts().to_dict())
    return df_summary


def outlier_summary(input_df, iqr_mul = 3):
    out_dict = {}
    for i in list(input_df.select_dtypes(include=[np.number]).columns):
        if len(input_df[i].unique()) > 1:
            quartile_1, quartile_3 = np.nanpercentile(input_df[i], q = [25, 75])
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr * iqr_mul)
            upper_bound = quartile_3 + (iqr * iqr_mul)
            pct_outliers_iqrscore = len(input_df[(input_df[i] > upper_bound) | (input_df[i] < lower_bound)]) / len(input_df)
            pct_outliers_iqrscore = np.round(pct_outliers_iqrscore*100, 4)
            pct_outliers_zscore = len(input_df[(np.abs(stats.zscore(input_df[i])) > 3)])/len(input_df[i])
            pct_outliers_zscore = np.round(pct_outliers_zscore*100, 4)
            out_dict[i] = {'pct_outliers_iqrscore': pct_outliers_iqrscore, 'pct_outliers_zscore': pct_outliers_zscore}
    return pd.DataFrame(out_dict).T


def remove_outliers(input_df, check_cols, method):
    inlier_index = list(input_df.index)
    if method.lower() == 'z_score':
        output_df = input_df[(np.abs(stats.zscore(input_df[check_cols])) <= 3).all(axis=1)]
    elif method.lower() == 'iqr_score':
        for i in check_cols:
            quartile_1, quartile_3 = np.nanpercentile(input_df[i], q = [25, 75])
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr * 3)
            upper_bound = quartile_3 + (iqr * 3)
            tmp_inlier_index = list(input_df[(input_df[i] > upper_bound) | (input_df[i] < lower_bound)].index)
            inlier_index = [v for v in inlier_index if v not in tmp_inlier_index]
        output_df = input_df[input_df.index.isin(inlier_index)]
    return output_df