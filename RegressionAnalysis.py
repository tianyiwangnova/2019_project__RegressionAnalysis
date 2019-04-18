class RegressionAnalysis:

    import pandas as pd
    import numpy as np
    import itertools
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style="darkgrid")
    sns.set(font_scale=1.5)

    def __init__(self,
                 input_data: pd.DataFrame,
                 model_type: str,
                 has_const: int,
                 metrics: str,
                 ad_channels: list = None,
                 transformation_rate: float = None,
                 p_value_threshold = 0.1
                 ):
        '''
        input_data: a pandas dataframe with the Xs and y
        model_type: 'linear-linear', 'log-linear', 'linear-log' or 'log-log'
        has_const: Does the linear regression model have constant variable? 1 for yes and 0 for no
        metrics: column name for the response variable in input_data; e.g. "sales"
        ad_channels: Optional; a list of ad_channels (if the class is used for marketing mix modeling); e.g. ['tv', 'radio', 'social']
        transformation_rate: ad spend discount rate for ad stock transformation
        p_value_threshold: this is only for plotting purpose :) For distinguish significant coefficients from non-significant coefficients

        if the user wants to do adstock transformation, we will do it at the initial stage
        '''
        self.transformation_rate = transformation_rate
        self.ad_channels = ad_channels

        if transformation_rate is not None and (transformation_rate > 0 and transformation_rate <= 1):
            self.input_data = self.adstock_transformation_table(input_data)
        else:
            self.input_data = input_data

        self.model_type = model_type
        self.has_const = has_const
        self.metrics = metrics
        self.p_value_threshold = p_value_threshold
        self.X = None
        self.y = None
        self.result = None

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

    def adstock_transformation_table(self, input_data):
        '''
        adstock transformation for all marketing spending columns; will change input_data;
        plot line chart for each variable
        '''
        import matplotlib.pyplot as plt
        result_table = input_data[self.ad_channels].apply(self.ad_stock_transformation, axis = 0)
        fig, axes = plt.subplots(len(self.ad_channels), 1, sharey=True)
        fig.subplots_adjust(hspace=0.5)
        fig.set_figheight(2 * len(self.ad_channels))
        fig.set_figwidth(6)
        for i, v in enumerate(self.ad_channels):
            axes[i].plot(result_table.index, result_table[v])
            axes[i].set_title(v)
        return result_table

    def regression_prep(self):
        '''
        get the X and y for simple linear regression

        RETURNS:
        X: a pandas dataframe of the X matrix. Will add a const column (a column of 1) if the user wants intercept in the regression model
        y: a pandas series
        '''
        self.X = self.input_data[[c for c in self.input_data.columns if c != self.metrics]]
        self.y = self.input_data[self.metrics]

        if self.model_type == "log-linear":
            self.y = np.log(self.input_data[self.metrics])
        if self.model_type == "linear-log":
            self.X = self.X.apply(np.log)
        if self.model_type == "log-log":
            self.y = np.log(self.input_data[self.metrics])
            self.X = self.X.apply(np.log)

        if self.has_const == 1:
            self.X['const'] = 1



    def main_model(self):
        '''
        Linear regression;
        Calculate P values for coefficients;
        Calculate R square
        :return:
        result: a dictionary with
            predictors: a pandas dataframe shows the fitted coefficient, t value and p value for each predictor (including the constant variable if applicable)
            y_hat: fitted y with this model
            r square
        '''
        from scipy import stats as stats
        #regession
        X_t = np.transpose(self.X)
        X_t_X = np.dot(X_t, self.X)
        X_t_X_inv = np.linalg.inv(X_t_X)
        df = len(self.y) - (self.X.shape[1]) #degree of freedom

        #estimators
        Beta = np.dot(np.dot(X_t_X_inv, X_t), self.y)

        # fitted y
        y_hat = np.dot(self.X, Beta)

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

        result = {'predictors': pd.DataFrame({'variable': self.X.columns,
                                              'coefficient': Beta,
                                              't_value': t_beta,
                                              'p_value': p_beta})
                                             .sort_values(by=['coefficient'], ascending=False),
                  'y_hat': y_hat,
                  'r_square': r_square}

        self.result = result

        return result

    # def predictor_coefficient_analysis(self):
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #     predictors = self.result['predictors']
    #     plt.rcParams['figure.figsize'] = ((len(predictors) - 1) * 2, int((len(predictors) - 1) * 1.6))
    #     predictors['p_value_range'] = np.where(predictors.p_value <= 0.05, '<=0.05', '>0.05')
    #     predictors['% increase in {}'.format(self.metrics)] = predictors['coefficient'] * 100
    #     ax = sns.barplot(x="variable",
    #                      y='% increase in {}'.format(self.metrics),
    #                      hue="p_value_range",
    #                      data=predictors[predictors.variable != 'const'],
    #                      dodge=False)
    #     ax.set_title('Percentage of metrics lifted by one unit increase of the predictor')









