# Last updated: 2019-05-02 

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.25)

from scipy import stats as stats

class RegressionAnalysis:

    def __init__(self,
                 input_data: pd.DataFrame,
                 model_type: str,
                 metrics: str,
                 ad_channels: list = None,
                 transformation_rate: float = None,
                 p_value_threshold = 0.1,
                 graph: bool = True
                 ):
        '''
        input_data: a pandas dataframe with the Xs and y
        model_type: 'linear-linear', 'log-linear', 'linear-log' or 'log-log'
        metrics: column name for the response variable in input_data; e.g. "sales"
        ad_channels: Optional; a list of ad_channels (if the class is used for marketing mix modeling); e.g. ['tv', 'radio', 'social']
        transformation_rate: ad spend discount rate for ad stock transformation
        p_value_threshold: this is only for plotting purpose :) For distinguishing significant coefficients from non-significant coefficients
        graph: whether to plot first step analysis charts when initialize the class
        X: X matrix for linear 
        y: y array for linear regressoin
        result: result for main_model
        plots: a dictionary with all the plots generated

        if the user wants to do adstock transformation, we will do it at the initial stage 
        '''
        self.transformation_rate = transformation_rate
        self.ad_channels = ad_channels
        self.plots = {}
        
        #process input_data
        if transformation_rate is not None and (transformation_rate > 0 and transformation_rate <= 1):
            self.input_data = self.adstock_transformation_table(input_data, graph)
        else:
            self.input_data = input_data

        self.model_type = model_type
        self.metrics = metrics
        self.p_value_threshold = p_value_threshold

        #process X and y for linear regression
        self.X, self.y = self.regression_prep(graph)
        self.result = None
        self.contribution_table = None

    def ad_stock_transformation(self, array: list) -> pd.Series:
        '''
        ad stock transformation for one array
        example: the investment we have for the first month will keep 80% of the effect in the second month and 80% * 80% in the 3rd and so on
        '''
        result = []
        current_stock = 0
        for v in array:
            current_stock = current_stock * self.transformation_rate + v
            result.append(current_stock)
        return pd.Series(result)

    def adstock_transformation_table(self, input_data, graph):
        '''
        adstock transformation for all marketing spending columns;
        plot line chart for each variable
        '''     
        result_table = input_data.copy() #copy the initial input dataframe
        result_table[self.ad_channels] = result_table[self.ad_channels].apply(self.ad_stock_transformation, axis = 0) #adstock transformation on the ad channel spendings
        
        #plot line charts for each channel to see how the ad effect accumulates
        if graph:
            fig, axes = plt.subplots(len(self.ad_channels), 1, sharey=True)
            fig.subplots_adjust(hspace=0.5)
            fig.set_figheight(2 * len(self.ad_channels))
            fig.set_figwidth(6)
            for i, v in enumerate(self.ad_channels):
                axes[i].plot(result_table.index, result_table[v])
                axes[i].set_title(v)
            self.plots['adstock_line_chart'] = fig
        return result_table

    def regression_prep(self, graph):
        '''
        get the X and y for simple linear regression

        RETURNS:
        X: a pandas dataframe of the X matrix. Will add a const column (a column of 1) 
        y: a pandas series
        '''
        X = self.input_data[[c for c in self.input_data.columns if c != self.metrics]]
        y = self.input_data[self.metrics]

        if self.model_type == "log-linear":
            y = np.log(self.input_data[self.metrics])
        if self.model_type == "linear-log":
            X = X.apply(np.log)
        if self.model_type == "log-log":
            y = np.log(self.input_data[self.metrics])
            X = X.apply(np.log)

        X['const'] = 1
        
        if graph:
            X1 = X.copy().drop('const', axis = 1)
            X1[self.metrics] = y
            p = sns.pairplot(X1, diag_kind="kde", kind="reg")
            self.plots['X_y_pairplots'] = p

        return X, y

    def main_model(self, X_custom = None, y_custom = None, graph = True, rewrite_result = True):
        '''
        Linear regression;
        Calculate P values for coefficients;
        Calculate R square

        Users can pass in other Xs and y that are different from the original one they feed into the class during init
        
        RETURNS
        result: a dictionary with
            predictors: a pandas dataframe shows the fitted coefficient, t value and p value for each predictor (including the constant variable)
            y_hat: fitted y with this model
            r square
        '''       
        X = self.X if X_custom is None else X_custom
        y = self.y if y_custom is None else y_custom

        #regression process...
        X_t = np.transpose(X)
        X_t_X = np.dot(X_t, X)
        X_t_X_inv = np.linalg.inv(X_t_X)
        df = len(self.y) - (X.shape[1]) #degree of freedom

        #estimators
        Beta = np.dot(np.dot(X_t_X_inv, X_t), self.y)

        # fitted y
        y_hat = np.dot(X, Beta)

        # sum of squares --- residuals
        ss_res = sum(np.square(self.y - y_hat))

        # calculate p_value for estimators
        mse = ss_res / df
        var_beta = mse * X_t_X_inv.diagonal()
        t_beta = Beta / np.sqrt(var_beta)
        p_beta = [(1 - stats.t.cdf(np.abs(t), df)) * 2 for t in t_beta]

        # calculate R square
        ss_total = np.var(list(self.y)) * len(self.y)
        r_square = (ss_total - ss_res) / ss_total

        result = {'predictors': pd.DataFrame({'variable': X.columns,
                                              'coefficient': Beta,
                                              't_value': t_beta,
                                              'p_value': p_beta})
                                             .sort_values(by=['coefficient'], ascending=False),
                  'y_hat': y_hat,
                  'r_square': r_square}

        if rewrite_result:
            self.result = result

        if graph:
            self.plots['coefficient_analysis'] = self.predictor_coefficient_analysis()

        return result

    def predictor_coefficient_analysis(self):
        '''
        plot the coefficents of each predictor colored by the level of significance
        '''
        predictors = self.result['predictors'].copy()
        plt.rcParams['figure.figsize'] = ((len(predictors) - 1) * 2, int((len(predictors) - 1) * 1.6))
        predictors['p_value_range'] = np.where(predictors.p_value <= self.p_value_threshold, '<={}'.format(self.p_value_threshold), '>{}'.format(self.p_value_threshold))

        if self.model_type == "linear-linear":
            predictors['units increased in {}'.format(self.metrics)] = predictors['coefficient']
            ax = sns.barplot(x="variable",
                         y='units increased in {}'.format(self.metrics),
                         hue="p_value_range",
                         data=predictors[predictors.variable != 'const'],
                         dodge=False)
            ax.set_title('Units increased in {} by one unit increase in the predictor'.format(self.metrics))

        if self.model_type == "log-linear":
            predictors['% increased in {}'.format(self.metrics)] = predictors['coefficient'] * 100
            ax = sns.barplot(x="variable",
                         y='% increased in {}'.format(self.metrics),
                         hue="p_value_range",
                         data=predictors[predictors.variable != 'const'],
                         dodge=False)
            ax.set_title('% increased in {} by one unit increase in the predictor'.format(self.metrics))
       
        if self.model_type == "linear-log":
            predictors['units increased in {}'.format(self.metrics)] = predictors['coefficient'] * 100
            ax = sns.barplot(x="variable",
                         y='units increased in {}'.format(self.metrics),
                         hue="p_value_range",
                         data=predictors[predictors.variable != 'const'],
                         dodge=False)
            ax.set_title('Units increased in {} by 1% increase in the predictor'.format(self.metrics))

        if self.model_type == "log-log":
            predictors['% increased in {}'.format(self.metrics)] = predictors['coefficient'] * 100
            ax = sns.barplot(x="variable",
                         y='% increased in {}'.format(self.metrics),
                         hue="p_value_range",
                         data=predictors[predictors.variable != 'const'],
                         dodge=False)
            ax.set_title('% increased in {} by 1% increase in the predictor'.format(self.metrics))

        return ax

    def fit(self, x):
        '''
        predict metrics based on the new X fed
        '''
        coef_dict = {}
        predictors = self.result['predictors'].copy()
        for i in range(len(predictors)):
            coef_dict[predictors.loc[i, 'variable']] = predictors.loc[i, 'coefficient']
        y = coef_dict['const']
        for v in x:
            y = y + x[v] * coef_dict[v]
        return y


    def calc_seq_r_square(self, variable_sets, variable_to_drop):
        '''
        calculate sequential r square; part of the contribution analysis

        variable sets: the larger variable sets; should be a list of strings
        variable_to_drop: the smaller variable sets; should be a list of strings
        '''
        r_square_large = self.main_model(X_custom = self.X[variable_sets], graph = False, rewrite_result = False)['r_square']
        r_square_small = self.main_model(X_custom = self.X[variable_sets].drop(variable_to_drop, axis = 1), graph = False, rewrite_result = False)['r_square']
        seq_r_square = r_square_large - r_square_small
        return seq_r_square

    def calc_lmg(self, variable_to_drop):
        '''
        calculate lmg (a predictor importance metrics)

        variable_to_drop: a string
        '''
        seq_r_square_all = [self.main_model(X_custom = self.X[[variable_to_drop, 'const']], graph = False, rewrite_result = False)['r_square']] 
        for i in range(1, len(self.X.columns)-1):
            sum_seq_r_current = 0
            predictor_sets_all = list(itertools.combinations(set([j for j in self.X.columns if j not in [variable_to_drop, 'const']]), i))
            for subset in predictor_sets_all:
                subset = list(subset) + [variable_to_drop, 'const'] 
                seq_r_square = self.calc_seq_r_square(subset, variable_to_drop)
                sum_seq_r_current = sum_seq_r_current + seq_r_square
            seq_r_square_all.append(sum_seq_r_current / len(predictor_sets_all))
        lmg = sum(seq_r_square_all) / (len(self.X.columns) - 1)

        return lmg

    def y_process(self, y_value):
        '''
        transform the fitted y from the model to the real y value we care in real life
        '''
        if self.model_type in ["linear-linear", "linear-log"]:
            y_value = y_value
        if self.model_type in ["log-linear", "log-log"]:
            y_value = np.exp(y_value)
        return y_value


    def calc_contribution(self, graph = True):
    
        initial_contribution = {}
        adjusted_contribution = {}
        coefficient_dict = {}

        predictors = self.result['predictors'].copy()

        for i in range(len(predictors)):
            coefficient_dict[predictors.loc[i, 'variable']] = predictors.loc[i, 'coefficient']
        
        for v in self.X:
            initial_contribution[v] = self.y_process(self.X[v] * coefficient_dict[v])
       
        difference = self.y_process(self.y) - self.y_process(self.result['y_hat'])
        
        # calculate relative importance for variables
        contribution_to_diff = {}
        
        for v in self.X.columns:
            if v == 'const':
                contribution_to_diff['const'] = (1 - self.result['r_square']) * coefficient_dict['const'] * np.ones(len(self.X))
            else:
                contribution_to_diff[v] = self.calc_lmg(v) * coefficient_dict[v] * np.array(self.X[v])
        
        contribution_to_diff = pd.DataFrame(contribution_to_diff)

        contribution_to_diff = contribution_to_diff.div(contribution_to_diff.sum(1), 0)
       
        # allocate difference to initial contributions
        for v in self.X:
            adjusted_contribution[v] = initial_contribution[v] + difference * contribution_to_diff[v]
        
        contribution_table = pd.DataFrame(adjusted_contribution)
        self.contribution_table = contribution_table
        
        if graph:
            self.plots['contribution_chart'] = self.contribution_chart()

        return contribution_table

    def contribution_chart(self):
        '''
        plot contribution chart (stacked bar chart)
        '''
        contribution_table = self.contribution_table.copy()
        columns_order = ['const'] + [c for c in contribution_table.columns if c != 'const']
        contribution_table = contribution_table[columns_order]

        plt.figure(figsize = (20,10))
        
        plt.clf()
        params = {'legend.fontsize': 20, 'legend.handlelength': 1}
        plt.rcParams.update(params)
        
        positive_bottom = pd.Series([0] * len(contribution_table))
        negative_bottom = pd.Series([0] * len(contribution_table))

        for i in range(len(contribution_table.columns)):

            current_series = contribution_table.iloc[:, i]
            current_bottom = positive_bottom
            negative_indexes = current_series[current_series < 0].index.values

            if len(negative_indexes) > 0:
              current_bottom[negative_indexes] = negative_bottom[negative_indexes]

            plt.bar(contribution_table.index.values, current_series, bottom = current_bottom, label = contribution_table.columns[i])

            positive_bottom = positive_bottom + np.where(current_series > 0, current_series, 0)
            negative_bottom = negative_bottom + np.where(current_series < 0, current_series, 0)
  
        plt.legend()
        plt.title("Each predictor's contribution to the {}".format(self.metrics))

    def ROI(self):
        '''
        calculate the ROI of each marketing channel;
        ROI = contribution to metrics / total investment in that channel
        '''
        ad = self.input_data[self.ad_channels].copy()
        ad_total = pd.DataFrame(ad.apply(sum, axis = 0)).rename({0:"total_spend"}, axis = 1)
        contribution_total = pd.DataFrame(self.contribution_table[self.ad_channels].apply(sum, axis = 0)).rename({0:"{}_contributed".format(self.metrics)}, axis = 1)
        roi = ad_total.join(contribution_total).reset_index().rename({"index":"channel"}, axis = 1)
        roi['ROI'] = roi["{}_contributed".format(self.metrics)] / roi['total_spend']

        self.roi = roi.sort_values(by = ['ROI'])

        ax = sns.barplot(x = "channel", y = "ROI", data = roi)
        ax.set_title("return of investment of different channels")
        self.plots['roi'] = ax

        return roi


