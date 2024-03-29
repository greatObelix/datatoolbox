# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np


class minh_dataXplore:
#==============================================================================
#     constructor: create dataframe
#==============================================================================
    def __init__(self, csv=None,df=None, summary=1):
        if csv is not None:
            self.df = pd.read_csv(csv)
        else:
            self.df = df
        
        if (summary):
            # sample 3lines
            print(self.df.head(3))
            print(self.df.describe())
        
#==============================================================================
#     data exploration
#==============================================================================
    # get unique values count for a column
    def uniqvalue(self,col):
        print (self.df[col].value_counts(),'\n')
        return
    
    # get summary of missing value from row/column
    def missing_count(self,x):
        return sum(x.isnull())
        
    def missing_summary(self, axis):
        # axis=0 defines that function is to be applied on each column
        # axis=1 defines that function is to be applied on each row
        print ('Missing Value per : ')
        print(self.df.apply(self.missing_count,axis=axis))         
        return
    
    def histogram(self, col, numbin=50):
        self.df[col].hist(bins=numbin)
        return
        
    def boxplot(self, col, by=None):
        self.df.boxplot(column=col, by=by, return_type='axes')
        return
    
    # scatter plot all columns against each other
    def distroXplore(self, a=0.2, size=(20,20), diag='kde', col=[]):
        from pandas.tools.plotting import scatter_matrix        
        scatter_matrix(self.df[col], alpha =a, figsize=size, diagonal=diag)
        return        
        
    # pivot table where index = column to be used as row in pivot table could be array
    # column = column to be used as column in pivot table, array accepted
    # aggrfunc = function to aggregate
    # value = column to be aggregated
    def pivotTable(self,val,ind, col, aggfcn=np.sum)  :
        temp = self.df.pivot_table(values=val, index=ind, columns=col, aggfunc=aggfcn)
        print(temp)
        return
        
#==============================================================================
#     # clean up the data by
#     # replace NULL, normalize various spelling for the same word
#     # substitute words for numeric
#==============================================================================
    def fillNULL(self,col, value):
        self.df[col].fillna(value, inplace=True)
        
    # encode from one value to another, use for convert customized categorical to numeric
    # or to nomalized different spelling of same words       
    def encoding(self, col, dictcode):

        for key, value in dictcode.items():
            self.df[col].replace(key,value,inplace=True)
        return
    
    # auto convert categorical to numeric
    def auto_alpha2num(self, col):
        from sklearn.preprocessing import LabelEncoder
        
        le = LabelEncoder()
        for i in col:
            self.df[i] = le.fit_transform(self.df[i])
        return 

    # one hot encoding
    def onehotencod(self, header):
        self.df = pd.get_dummies(self.df)
        
#==============================================================================
#   transform data, log and scale
#==============================================================================        
    #apply a logarithmic transformation on the data so that the very large and
    #very small values do not negatively affect the performance of a learning algorithm
    # log of 0 is undefined, so need to map to +1, work for positive values
    def takelog(self,header):
        self.df[header] = self.df[header].apply(lambda x: np.log(x+1))

    # Applying a scaling to the data does not change the shape of each feature's distribution
    # normalization ensures that each feature is treated equally when applying supervised learners        
    def scale(self, numerical_header):
        from sklearn.preprocessing import MinMaxScaler

        # Initialize a scaler, then apply it to the features
        scaler = MinMaxScaler() # default=(0, 1)
        self.df[numerical_header] = scaler.fit_transform(self.df[numerical_header])
        
        
if __name__ == '__main__':
    print('\nmain dataXplore\n')
    test = minh_dataXplore('bigmartIII\Train_UWu5bXk.csv', summary=0)
    test.uniqvalue('Item_Fat_Content')
    test.missing_summary(axis=0)
    #test.histogram(col='Item_Outlet_Sales')
    #test.boxplot('Item_Outlet_Sales',by='Outlet_Location_Type')
    test.distroXplore(col=['Item_Weight','Item_Visibility','Item_Outlet_Sales']);
    test.pivotTable(val='Item_Outlet_Sales',col='Outlet_Type',ind='Outlet_Identifier')
    test.encoding(col='Item_Fat_Content',dictcode={'LF':'Low Fat','low fat':'Low Fat','Regular':'reg'})
    test.uniqvalue('Item_Fat_Content')
    test.auto_alpha2num(['Item_Fat_Content'])
    test.uniqvalue('Item_Fat_Content')
