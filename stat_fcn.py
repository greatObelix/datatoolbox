# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:02:40 2017

@author: mh636c
"""

# Z-score table lookup
#return z given probability
#return probablity given z
import scipy.stats as st
st.norm.ppf(.95)   #out 1.6448536269514722
st.norm.cdf(1.64)  #out 0.94949741652589625

#calculate zscore from array
a = np.array([ 0.7972,  0.0767,  0.4383,  0.7866,  0.8091,  0.1954, 0.6307, 0.6599,  0.1065,  0.0508])
import scipy.stats
stats.zscore(a)
#calc zscoaredf = 
pd.DataFrame(np.random.randint(100, 200, size=(5, 3)), columns=['A', 'B', 'C'])
pd.apply(stats.zscore)
#or just apply numeric col
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols].apply(zscore)

#coefficient of determination, R^2
#http://stattrek.com/statistics/dictionary.aspx?definition=coefficient_of_determination
#the square of the correlation (r) between predicted y scores and actual y scores
from sklearn.metrics import r2_score
r2_score(y_true,y_predict)
# f1 score where beta =0.5
from sklearn.metrics import fbeta_score
result = fbeta_score(y_test,predictions_test,beta=0.5)
# accuracy score
from sklearn.metrics import accuracy_score
result = accuracy_score(y_test,predictions_test)