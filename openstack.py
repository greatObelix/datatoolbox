# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:17:17 2016


"""

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # get raw data
    df = pd.read_csv('data.csv')
       
    # make copy of dataframe else it will complain modifying the copied df
    df_train = df[['VolumeDeletionAttempt','InstanceDeletionAttempt','RebootAttempt','InstanceCreationAttempt','VolumeCreationAttempt','percent.kernel.all.cpu.user']].copy()
    df_train['total_att'] = df_train['VolumeDeletionAttempt']+df_train['InstanceDeletionAttempt']+df_train['RebootAttempt']+df_train['InstanceCreationAttempt']+df_train['VolumeCreationAttempt']

    # set first 120 rows for training, and the remaining for testing
    # divide by 300 to get per second since data is 5min interval
    x_train = df_train.iloc[0:120,6]/300
    y_train = df_train.iloc[0:120,5]
    x_test = df_train.iloc[120:,6]/300
    
    
    # Create linear regression object
    linear = linear_model.LinearRegression()
    # Train the model using the training sets and check score
    # if the x_train is one dimension array or single feature use dataframe.reshape(-1,1)
    linear.fit(x_train.reshape(-1,1), y_train)
    print('Linear fit score: ',linear.score(x_train.reshape(-1,1), y_train))
    #Equation coefficient and Intercept
    print('Coefficient: \n', linear.coef_)
    print('Intercept: \n', linear.intercept_)
    # Predict on test data
    predicted = linear.predict(x_test.reshape(-1,1)) 

    # Output, need to convert to Series format to combine two arrays into a dataframe
    # reset index needed to align from 0, drop = True so that dataframe won't input row name from previous into new dataframe
    df_out=pd.concat([pd.Series(x_test).reset_index(drop=True),pd.Series(predicted).reset_index(drop=True)],axis=1)    
    df_out.rename(columns={0:'cpu'},inplace=True)
    #df_out.to_csv('predict.csv', index=False)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_train,y_train)
    ax.plot(x_train,linear.coef_[0]*x_train +linear.intercept_,color='red')
    plt.show()
