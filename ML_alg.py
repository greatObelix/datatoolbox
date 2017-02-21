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
    #Predict Output
    predicted= model.predict(x_test)
    
def k_nearest_neighbor():
    #Import Library
    from sklearn.neighbors import KNeighborsClassifier
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create KNeighbors classifier object model 
    KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5
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

def randomeforest():
    #Import Library
    from sklearn.ensemble import RandomForestClassifier #use RandomForestRegressor for regression
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create Random Forest object
    model= RandomForestClassifier()
    # Train the model using the training sets and check score
    model.fit(X, y)
    #Predict Output
    predicted= model.predict(x_test)
    
def dimReduction():
    # Dimensionality Reduction Alg
    #Import Library
    from sklearn import decomposition
    #Assumed you have training and test data set as train and test
    # Create PCA obeject pca= decomposition.PCA(n_components=k) #default value of k =min(n_sample, n_features)
    # For Factor analysis
    #fa= decomposition.FactorAnalysis()
    # Reduced the dimension of training dataset using PCA
    train_reduced = pca.fit_transform(train)
    #Reduced the dimension of test dataset
    test_reduced = pca.transform(test)
    #For more detail on this, please refer http://scikit-learn.org/stable/modules/decomposition.html#decompositions

def GradientBoosting():
    #Import Library
    from sklearn.ensemble import GradientBoostingClassifier
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create Gradient Boosting Classifier object
    model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    # Train the model using the training sets and check score
    model.fit(X, y)
    #Predict Output
    predicted= model.predict(x_test)

def GBM():
    #import libraries
    from sklearn.ensemble import GradientBoostingClassifier #for classification
    from sklearn.ensemble import GradientBoostingRegressor #for regression
    #use GBM function
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
    clf.fit(X_train, y_train)