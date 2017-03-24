# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:47:02 2016

@author: mh636c
"""

import matplotlib.pyplot as plt
import pandas as pd

def add_trendline(df):
    # sklearn example
    from sklearn import linear_model
    
    model = linear_model.LinearRegression()
    x_train = df[['mem_used']].reset_index(drop=True)
    y_train = df[['in_bytes']].reset_index(drop=True)
    model.fit(x_train,y_train)
    print("score:" + str(model.score(x_train,y_train)))
    print("coefficient: ",model.coef_)
    print("y-intercept", model.intercept_)
    predict = model.predict(x_train)
    
    #output trend graph, draw from original dataframe, save to a figure object
    #create new dataframe with trend datapts, call plot() and pass the figure objext
    # .flatten() to convert from numpy array to series
    trend_df = pd.DataFrame(data={'trend':predict.flatten()},index=x_train)
    ax = df.plot.scatter(x='mem_used',y='in_bytes')
    trend_df.plot(kind='line',ax=ax,colormap='RdBu')

    #np.polyfit() example
    #Needed to use .loc[] so single column becomes a pd.Series.
    #Selecting with [[]] keeps a single column as DataFrame, hence the dimension warning
    z = np.polyfit(x=df.loc[:, 'mem_used'], y=df.loc[:, 'in_bytes'], deg=1)
    p = np.poly1d(z)
    trend_df = pd.DataFrame(data={'trend':p(df.loc[:, 'mem_used']).flatten()},index=df.loc[:, 'mem_used'])
    ax = df.plot.scatter(x='mem_used', y='in_bytes')
    trend_df.plot(kind='line',ax=ax,colormap='RdBu')

def scatter_custom_xlabel():
    # define custom xlabels and major tick spacing
    # since scatter graph can only graph numeric value for x, the trick is to
    # first graph with some sequence of number e.g. range(0,100)
    # then after plot, replace the tick label with the dates
    # graph completely with matplotlib
    # vr_df is a pandas dataframe

    fig = plt.figure()
    
    # max # of major ticks
    max_ticks = 10
    # prep the list of labels, convert string 201702221540 into datetime as 2017-02-22 15:40:00
    # to replace the tick label later
    labels = []
    step = int(len(vr_df['timestamp'])/max_ticks)
    labels.append(dt.datetime.strptime(str(vr_df.iloc[0]['timestamp']),'%Y%m%d%H%M'))
    for i in range(1,max_ticks):
        if ((i*step) > len(vr_df['timestamp'])):
            break
        labels.append(dt.datetime.strptime(str(vr_df.iloc[i*step]['timestamp']),'%Y%m%d%H%M'))
    
    # first graph, set x-tick-labels, y-grid, and reduce empty space margin
    ax = fig.add_subplot(1,1,1)
    ax.set_xticklabels(labels)
    ax.yaxis.grid()
    ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
    ax.margins(x=0)
    ax.set_ylabel('active_flows')
    # graph with x as a a sequence of number
    ax.scatter(range(vr_df.index.size), vr_df['active_flows'], label='observations')    
    
    ax2 = fig.add_subplot(1,1,1)
    ax2.plot(range(len(vr_df['timestamp'])),vr_df['active_flows_mean'], 'r',label='predicted_mean')
    
    ax3 = fig.add_subplot(1,1,1)
    ax2.plot(range(len(vr_df['timestamp'])),vr_df['outliner_threshold'], 'g',label='outliner_threshold')
    
    plt.legend()
    fig.set_size_inches(15,10)
    fig.autofmt_xdate()

def scatter_timeseries():
    # graph scatter using the x-axis as dates in a timeseries
    # adjust graph format for very pretty datetime display
    # graph use dataframe plot()
    import datetime as dt
    import matplotlib.dates as dates
    # first convert the timestamp into internal representation of datetimes for the x-axis because
    # can't do scatter plot with x value as datetime type, first convert to datetime->str->datestr2num for 
    # internal representation of date in numbber for matplotlib
    # vr_df and anomaly_df are pandas dataframes
    # map() vs. apply(): map() is used on each element of the dataframe, apply() to entire row or entire column
    vr_df['datetime'] = vr_df['timestamp'].map(lambda x: dates.datestr2num(str(dt.datetime.strptime(str(x),'%Y%m%d%H%M'))))

    ax = vr_df.plot.scatter('datetime','active_flows',xticks=vr_df['timestamp'])
    # adjust graph format for very pretty datetime
    ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=(0),interval=1))
    ax.xaxis.set_minor_formatter(dates.DateFormatter('%d\n%a'))
    ax.xaxis.grid(True, which="minor")
    ax.yaxis.grid()
    ax.xaxis.set_major_locator(dates.MonthLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('\n\n\n%b\n%Y'))
    """
    # adjust graph x axis label for pretty day hour
    ax.xaxis.set_major_locator(dates.DayLocator())
    ax.xaxis.set_minor_formatter(dates.DateFormatter('%H'))
    ax.xaxis.set_minor_locator(dates.HourLocator(np.arange(0, 25, 6)))
    ax.xaxis.set_major_formatter(dates.DateFormatter('\n%a'))
    fig.autofmt_xdate()
    fig.set_size_inches(15,10)
    """
    # create anomaly dataframe
    x = vr_df[['datetime']].reset_index(drop=True)
    y = vr_df[['active_flows_mean']].reset_index(drop=True)
    outliner = vr_df[['outliner_threshold']].reset_index(drop=True)
    anomaly_df = pd.concat([x,y,outliner],axis=1)
    anomaly_df.set_index('datetime',inplace=True)
    anomaly_df.rename(columns={'active_flows_mean':'predicted_mean'},inplace=True)
    # pass ax in as parameter, so graph anomaly on the same graph as the vr_df data
    anomaly_df.plot(kind='line',colormap='PiYG', ax=ax,figsize=(15,10))
    plt.savefig('anomaly_fig.png')

def histogram(df):
    # historgram
    fig=plt.figure() #Plots in matplotlib reside within a figure object, use plt.figure to create new figure
    #Create one or more subplots using add_subplot, because you can't create blank figure
    ax = fig.add_subplot(1,1,1)
    #Variable
    ax.hist(df['speed_value'],bins = 10) # Here you can play with number of bins
    #Labels and Title
    plt.title('Speed distribution')
    plt.xlabel('Speed')
    plt.ylabel('#Port')
    plt.show()

def boxplot(df):
    fig=plt.figure()
    ax = fig.add_subplot(1,1,1)
    #Variable
    ax.boxplot(df['speed_value'])
    plt.show()
    
    # or seaborn
    #sns.boxplot(x='speed_value',y='95thPctle_InUtil%',data=df)    

def barchart(df):
    var = df.groupby('vnf_name').speed_value.sum()
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('VNF')
    ax1.set_ylabel('Commited Info Rate')
    ax1.set_title("Commited Info Rate per Customer")
    var.plot(kind='bar')

def linechart(df):
    var = df.groupby('vnf_name').speed_value.sum()
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_xlabel('VNF')
    ax1.set_ylabel('Commited Info Rate')
    ax1.set_title("Commited Info Rate per Customer")
    var.plot(kind='line')

def stackedCol(df):
    var = df.groupby(['BMI','Gender']).Sales.sum()
    var.unstack().plot(kind='bar',stacked=True,  color=['red','blue'], grid=False)

def scatter_basic(df):
    # matplot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(df['Age'],df['Sales']) #You can also add more variables here to represent color and size.
    plt.show()
    
    #seabornplot
    #sns.jointplot(df['speed_value'], y=df['95thPctle_InUtil%'], kind='scatter')

def bubble(df):    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(df['Age'],df['Sales'], s=df['Income']) # Added third variable income as size of the bubble
    plt.show()

def pie(df):
    ar=df.groupby(['Gender']).sum().stack()
    temp=var.unstack()
    type(temp)
    x_list = temp['Sales']
    label_list = temp.index
    pyplot.axis("equal") #The pie chart is oval by default. To make it a circle use pyplot.axis("equal")
    #To show the percentage of each pie slice, pass an output format to the autopctparameter 
    plt.pie(x_list,labels=label_list,autopct="%1.1f%%") 
    plt.title("Pastafarianism expenses")
    plt.show()

def violinplot(df):
    import seaborn as sns 
    sns.violinplot(df['vpe_id'], df['speed_value']) #Variable Plot
    sns.despine()

def hexbin(df):
    xmin = df['95thPctle_InMbps'].min()
    xmax = df['95thPctle_InMbps'].max()
    ymin = df['95thPctle_InUtil%'].min()
    ymax = df['95thPctle_InUtil%'].max()
    
    #plt.subplots_adjust(hspace=0.5)
    #plt.subplot(121)
    plt.hexbin(df['95thPctle_InMbps'], df['95thPctle_InUtil%'], bins='log', cmap=plt.cm.terrain_r)# plt.cm.YlOrRd_r)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title("Hexagon binning")
    cb = plt.colorbar()
    cb.set_label('counts')
    plt.show()
    
    
if __name__ == '__main__':

    df=pd.read_csv("test2.csv")
    #histogram(df)    
    #boxplot(df)
    #barchart(df)
    #linechart(df)
    violinplot(df)
    