# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:38:48 2016

@author: mh636c
"""

def linear_regr():
    #Import Library
    #Import other necessary libraries like pandas, numpy...
    from sklearn import linear_model
    #Load Train and Test datasets
    #Identify feature and response variable(s) and values must be numeric and numpy arrays
    # if the x_train is an single array or single featuer use dataframe.reshape(-1,1)
    x_train=input_variables_values_training_datasets
    y_train=target_variables_values_training_datasets
    x_test=input_variables_values_test_datasets
    # Create linear regression object
    linear = linear_model.LinearRegression()
    # Train the model using the training sets and check score
    linear.fit(x_train, y_train)
    linear.score(x_train, y_train)
    #Equation coefficient and Intercept
    print('Coefficient: \n', linear.coef_)
    print('Intercept: \n', linear.intercept_)
    #Predict Output
    predicted= linear.predict(x_test)
    
def logistic_regression():
    #Import Library
    from sklearn.linear_model import LogisticRegression
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create logistic regression object
    model = LogisticRegression()
    # Train the model using the training sets and check score
    model.fit(X, y)
    # R^2 score
    model.score(X, y)
    #Equation coefficient and Intercept
    print('Coefficient: \n', model.coef_)
    print('Intercept: \n', model.intercept_)
    #Predict Output
    predicted= model.predict(x_test)
    
def decision_tree():
    #Import Library
    #Import other necessary libraries like pandas, numpy...
    from sklearn import tree
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create tree object 
    model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
    # model = tree.DecisionTreeRegressor() for regression
    # Train the model using the training sets and check score
    model.fit(X, y)
    # R^2 score
    model.score(X, y)
    #Predict Output
    predicted= model.predict(x_test)
    
def support_vector_machine():
    #Import Library
    from sklearn import svm
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create SVM classification object 
    model = svm.svc() # there is various option associated with it, this is simple for classification. You can refer link, for mo# re detail.
    # Train the model using the training sets and check score
    model.fit(X, y)
    # R^2 score
    model.score(X, y)
    #Predict Output
    predicted= model.predict(x_test)

def naive_bayes():
    #Import Library
    from sklearn.naive_bayes import GaussianNB
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create SVM classification object model = GaussianNB() # there is other distribution for multinomial classes like Bernoulli Naive Bayes, Refer link
    # Train the model using the training sets and check score
    model.fit(X, y)
    # R^2 score
    model.score(X, y)    
    #Predict Output
    predicted= model.predict(x_test)
    
def k_nearest_neighbor():
    #Import Library
    from sklearn.neighbors import KNeighborsClassifier
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create KNeighbors classifier object model 
    model = KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
    # Train the model using the training sets and check score
    model.fit(X, y)
    #Predict Output
    predicted= model.predict(x_test)
    
def kmean():
    #Import Library
    from sklearn.cluster import KMeans
    #Assumed you have, X (attributes) for training data set and x_test(attributes) of test_dataset
    # Create KNeighbors classifier object model 
    k_means = KMeans(n_clusters=3, random_state=0)
    # Train the model using the training sets and check score
    model.fit(X)
    #Predict Output
    predicted= model.predict(x_test)

def randomforest():
    #Import Library
    from sklearn.ensemble import RandomForestClassifier #use RandomForestRegressor for regression
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create Random Forest object
    model= RandomForestClassifier()
    # Train the model using the training sets and check score
    model.fit(X, y)
    # R^2 score
    model.score(X, y)    
    #Predict Output
    predicted= model.predict(x_test)
    
def dimReduction():
    # Dimensionality Reduction Alg
    #Import Library
    from sklearn import decomposition
    #Assumed you have training and test data set as train and test
    # Create PCA obeject 
    pca= decomposition.PCA(n_components=k) #default value of k =min(n_sample, n_features)
    # For Factor analysis
    fa= decomposition.FactorAnalysis()
    # Reduced the dimension of training dataset using PCA
    train_reduced = pca.fit_transform(train)
    #Reduced the dimension of test dataset
    test_reduced = pca.transform(test)
    #For more detail on this, please refer http://scikit-learn.org/stable/modules/decomposition.html#decompositions

def AdaptiveBoosting():
    #Import Library
    from sklearn.ensemble import AdaBoostClassifier # or regression use AdaBoostRegressor
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create Gradient Boosting Classifier object
    clf= AdaBoostClassifier(random_state=0)
    # Train the model using the training sets and check score
    clf.fit(X, y)
    # R^2 score
    model.score(X, y)    
    #Predict Output
    predicted= clf.predict(x_test)

def GradientBoosting():
    #import libraries
    from sklearn.ensemble import GradientBoostingClassifier #for classification
    from sklearn.ensemble import GradientBoostingRegressor #for regression
    #use GBM function
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    
# split test validation
# split 80% train, 20% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

#grid serach
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
#Create a decision tree regressor object
regressor = DecisionTreeRegressor(random_state = 0)
#create a dictionary for the parameter 'max_depth' with a range from 1 to 10
params = {'max_depth':[x for x in range(1,11)]}
#Create the grid search object
grid = GridSearchCV(regressor,param_grid=params,scoring=scoring_fnc, cv=crossvalidation_sets)
# Fit the grid search object to the data to compute the optimal model
grid = grid.fit(X, y)
# Return the optimal model after fitting the data
optimal = grid.best_estimator_
#get best parameters
optimal.get_params()['max_depth']
#predict with best param
optimal.predict(data)
    
#k-fold validation
# k-fold is a type of cross validation where the data are divided into k bins. For each experiment, pick one of the k bins as the test set, 
#the remaining k-1 bins as training. Run k separate experiments and average all k test results. 
#This technique helps to test different part of the data to prevent overfitting 
#i.e. it prevents grid search from returning a parameter set that optimized specifically for a specific training data set but not overall.
from sklearn.model_selection import KFold
cv_set = KFold(n_splits=10)
for train_index, test_index in cv_sets.split(X):
     print("%s %s" % (train_index, test_index))
#Shufflesplit
#ShuffleSplit() for an alternative form of cross-validation (see the 'cv_sets' variable). 
#The ShuffleSplit() will create 10 ('n_splits') shuffled sets, and for each shuffle, 20% ('test_size') of the data will be used as the validation set.
from sklearn.model_selection import ShuffleSplit
cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
for train_index, test_index in cv_sets.split(X):
     print("%s %s" % (train_index, test_index))
     
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

# pipeline prediction
def pipeline_train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test,predictions_test)
    
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test,predictions_test,beta=0.5)
       
    # Success
    print ("{} trained on {} samples. train {:.3f}sec predict {:.3f}sec fsctest {:.3f}".format(learner.__class__.__name__, sample_size,results['train_time'],results['pred_time'],results['f_test']))
        
    # Return the results
    return results
#to use pipeline, example
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# TODO: Initialize the three models
clf_A = svm.SVC()
clf_B = RandomForestClassifier()
clf_C = AdaBoostClassifier()

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100
# HINT: samples_1 is 1% of samples_100
samples_100 = len(X_train)
samples_10 = int(0.1*samples_100)
samples_1 = int(0.01*samples_100)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        pipeline_train_predict(clf, samples, X_train, y_train, X_test, y_test)

    
