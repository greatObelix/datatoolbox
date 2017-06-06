# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:56:41 2017


"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
        
class timeseries():

    def __init__(self, filename):
        
        dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d%H%M%')
        #to read data as timeseries, 
        #pass in argument to parse time from string
        #parse_dates: specify column contain date-time info
        #index_col: dataframe index need to be datetime column
        #date_parser: fcn to convert input string to datetime variable. pandas default is YYYY-MM-DD HH:MM:SS need a parser if not in the same format
        self.df = pd.read_csv(filename,parse_dates=['timestamp'],index_col='timestamp',date_parser=dateparse)

    def test_stationarity(self):
        # Using Plotting Rolling Statistic and Dickey-Fuller Test        
        #Determing rolling statistics
        #window determin how many data points/observation per window
        # visually check if mean/stdev increase as fcn of time
        rolmean = pd.rolling_mean(self.df, window=287)
        rolstd = pd.rolling_std(self.df, window=287)
        #Plot rolling statistics:
        orig = plt.plot(self.df['in_tpkts'], color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)

        #Perform Dickey‐Fuller test
        # if Test Statistic > Critical Values, then not stationary 
        print ('Results of Dickey‐Fuller Test:')
        dftest = adfuller(self.df['in_tpkts'], autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p‐value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print (dfoutput)

    def make_stationary(self):
        # remove trend and seasonality 
        #for positive trend, to penalize higher values do log/squqreroot/cube root etc...
        self.ts_log = np.log(self.df)
        
        #estimate or model trend, then remove from the series. diff appraoches        
        # aggregation: take avg for monthly/weekly avg
        # smooth: taking rolling avg
        # poly fit : fit a regression model
        
        # Exanoke 1: using smoothing as example, rolling avg
        moving_avg = pd.rolling_mean(self.df,window=287)
        ts_log_moving_avg_diff = self.ts_log - moving_avg
        ts_log_moving_avg_diff.dropna(inplace=True)
        
        # Example 2: using exponential weighted moving avg (EWMA)
        # halflife is same as window, how many datapoint to make up 1 cycle
        expwighted_avg = pd.ewma(self.ts_log, halflife=287)
        ts_log_ewma_diff = self.ts_log - expwighted_avg
        
        # Example 3: differencing: take the difference of the observation at a particular instant 
        # with that at the previous instant
        self.ts_log_diff = self.ts_log - self.ts_log.shift()
        
        # Example 4: decomposing
        # trend and seasonality are modeled separately and the remaining part of the series is returned
        # pandas.DataFrame with index doesn't work, need to pass in numpy value as datafram.values
        decomposition = seasonal_decompose(ts_log.values, freq=288)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        
    def ACF_PACF_plot(self):
        #plot ACF and PACF to find the number of terms needed for the AR and MA in ARIMA
        # ACF finds MA(q): cut off after x lags 
        # and PACF finds AR (p): cut off after y lags 
        # in ARIMA(p,d,q) 
        lag_acf = acf(self.ts_log_diff, nlags=20)
        lag_pacf = pacf(self.ts_log_diff, nlags=20, method='ols')
        
        #Plot ACF:
        ax=plt.subplot(121)
        plt.plot(lag_acf)
        ax.set_xlim([0,5])
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y= -1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
        plt.axhline(y= 1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
        plt.title('Autocorrelation Function')
        
        #Plot PACF:
        plt.subplot(122)
        plt.plot(lag_pacf)
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y= -1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
        plt.title('Partial Autocorrelation Function')
        plt.tight_layout()
        
    def ARIMA_fit(self):
        # order=(p,d,q) AR and MA can also be modeled separately by enter 0 for either p or q
        model = ARIMA(ts_log, order=(5,1,5))
        self.results_ARIMA = model.fit(disp=-1)
        
        print(results_ARIMA.summary())
        
        plt.plot(ts_log_diff)
        plt.plot(results_ARIMA.fittedvalues, color='r')
        plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff['in_tpkts'])**2))

    def ARIMA_predict_fitted(self):
        # using ARIMA prediction fcn to predic modeled (more works involved than using the forecast fcn
        # and scale back to original data
        #1st: fitted will return the fitted values, but this is the difference between the lag
        predictions_ARIMA_diff = pd.Series(self.results_ARIMA.fittedvalues, copy=True)
        #2nd: find the cummulative sum (cumsum): The way to convert the
        # differencing to log scale is to add these differences consecutively to the base number. 
        # An easy way to do it is to frst determine the cumulative sum at index and then add it to the base number.
        predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
        #3rd: to add them to base number. For this create a series with all values as base
        # number and add/substract the differences to it
        predictions_ARIMA_log = pd.Series(ts_log.ix[0]['in_tpkts'], index=ts_log.index)
        predictions_ARIMA_log = predictions_ARIMA_log.subtract(predictions_ARIMA_diff_cumsum,fill_value=0)
        #4th: take exponent to scale the log back
        predictions_ARIMA = np.exp(predictions_ARIMA_log)

    def ARIMA_forecast(self):
        # out of sample forecast, forecast future date out of training data
        # using forecast fcn, less work, no need to find cumsum and substract original data
        # forecast sometime might not work, need to induce using old day of the same time period, 
        # e.g. Wednesday to Wednesday
        forecast_log = pd.Series(results_ARIMA.forecast(steps=288,exog=exog)[0],index=pd.date_range(start='2017-02-09 00:00:00', periods=288,freq='5min'))
        forecast = np.exp(forecast_log)
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(vr_df2_ts, label='original')
        ax2 = fig.add_subplot(1,1,1)
        ax2.plot(predictions_ARIMA,'r')
        ax3= fig.add_subplot(1,1,1)
        ax3.plot(forecast,'g')
        plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-vr_df2_ts['in_tpkts'])**2)/len(vr_df2_ts['in_tpkts'])))

        ax.xaxis.set_major_locator(dates.DayLocator())
        ax.xaxis.set_minor_formatter(dates.DateFormatter('%H'))
        ax.xaxis.set_minor_locator(dates.HourLocator(np.arange(0, 25, 6)))
        ax.xaxis.set_major_formatter(dates.DateFormatter('\n%a'))
        fig.autofmt_xdate()
        fig.set_size_inches(15,10)
        
    def ARIMA_forcast2(self):
        # this approach forecast 1 data pt at a time, then add the new forecast datapoint to the training data
        # then repeat
        import warnings
        warnings.filterwarnings('ignore')
        
        # test without taking log of data
        # using rolling avg 
        y = vr_df2_ts.values
        train = vr_df2_ts.values[286:574]
        prediction = list()
        for t in range(288):
            modelY = ARIMA(y, order=(1,1,1))
            results = modelY.fit(disp=-1)
            out = results.forecast()
            yhat = out[0]
            prediction.append(yhat)
            y = np.append(y,train[t])
            
        forecast = pd.Series(prediction,index=pd.date_range(start='2017-02-09 00:00:00', periods=288,freq='5min'))
        exog = vr_df2_ts.iloc[286:574]
        exog.set_index(pd.date_range(start='2017-02-09 00:00:00', periods=288,freq='5min'),inplace=True)
        
        plt.plot(vr_df2_ts)
        plt.plot(exog,'g')
        plt.plot(forecast,'r')

    # create a differenced series
    def difference(self, dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = float(dataset[i] - dataset[i - interval])
            diff.append(value)
        return np.array(diff)
    
    # invert differenced value
    def inverse_difference(self, history, yhat, interval=1):
        return yhat + history[-interval]
        
    # forecast3, out of sample forecast, first diff to eliminate seasonality, then invert the diff
    # by adding historical data by the ith datapoint define by cycle
    def ARIMA_forcast3(self):
        # load dataset
        series = pd.Series(vr_df['ACTIVE_FLOWS'][0:7000])
        # seasonal difference
        X = series.values
        cycle = 288 #2016
        differenced = difference(X, cycle)
        # fit model
        model = ARIMA(differenced, order=(1,1,1))
        model_fit = model.fit(disp=0)
        # multi-step out-of-sample forecast
        forecast = model_fit.forecast(steps=2016)[0]
        # invert the differenced forecast to something usable
        history = [x for x in X]
        step = 1
        forecast_values = []
        for yhat in forecast:
            inverted = inverse_difference(history, yhat, cycle)
            #print('Day %d: %f' % (day, inverted))
            forecast_values.append(inverted)
            history.append(vr_df['ACTIVE_FLOWS'][7000+step-1])
            step += 1

    # similar to ARIMA_forecast3 except forecast num_forecast points at a time
    # after each forecast refit the model with num_forecast points, repeat until end of actual data array use for test
    def ARIMA_forecast4(self):
        # parameters
        num_train_init = 7318 
        num_forecast = 12 #one day = 288 data points
        cycle = 288 #for a total 288 samples per day
        startdate = vr_df.index[num_train]
        field = 'DELETED_FLOWS'
        # array of predicted values
        forecast_values = []
        
        for i in range(0,int(len(vr_df)/num_forecast)):
            # check array for out of bound
            num_train_current = i*num_forecast+num_train_init
            if ((num_train_current) > len(vr_df)):
                break
            # load dataset
            series = pd.Series(vr_df[field][0:num_train_current])
            # Make data stationary: seasonal difference
            X = series.values
            differenced = difference(X, cycle)
            # fit model
            model = ARIMA(differenced, order=(1,1,1))
            model_fit = model.fit(disp=0)
            # multi-step out-of-sample forecast
            forecast = model_fit.forecast(steps=num_forecast)[0]
            # invert the differenced forecast to something usable
            history = [x for x in X]
            step = 1
            for yhat in forecast:
                inverted = inverse_difference(history, yhat, cycle)        
                forecast_values.append(inverted)
                #append actual data
                try:
                    history.append(vr_df[field][num_train_current+step-1])
                except:
                    # reached the end of actual data array, use forecasted values to estimate
                    history.append(inverted)
                step += 1
     
if __name__ == '__main__':
    
    rcParams['figure.figsize'] = 15, 6
    ts = timeseries('pkt_ts.csv')
    # 1st test for stationary
    ts.test_stationarity()
    # if fail, make stationary
    ts.make_stationary()
    # 2nd test again for stationary
    ts.test_stationarity()
    ts.ACF_PACF_plot()
    ts.ARIMA_fit()
    ts.ARIMA_forecast()
    

