import numpy as np
import pandas as pd

from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import PolynomialFeatures

# For optimal performance, set degree to 3. For speed, set to 1
polynomial_features = PolynomialFeatures(interaction_only=False, include_bias=True, degree=1)

from utilities import (convert_to_words, print_full, get_position_stats, combine_csv, resolve_acc_gyro,
                       blank_filter, concat_data, update_df)

def trial(df_train, test_data):
    """The trial is for running predictions on test data."""
    
    my_test_data = test_data.drop(['avg_stand'], axis=1)
    y = df_train['state'].values
    X = df_train.drop(['avg_stand', 'stand', 'state', 'index'], axis=1)
    if X.isnull().values.any() == False: 

        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=8, min_samples_split=4,
                min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)

        X = polynomial_features.fit_transform(X)

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)

    else: 
        print "Found NaN values"

    rf.fit(X_train, y_train)
    polynomial_test_data = polynomial_features.fit_transform(my_test_data)
    rf_pred2 = rf.predict(polynomial_test_data)
    print rf_pred2
    test_data['state'] = rf_pred2
    final_prediction = convert_to_words(rf_pred2)
    print_full(final_prediction)
    get_position_stats(final_prediction)
    return test_data


def trial_standup(df_train, test_data):
    """
    Test 1: 1s followed by 3s
    """
    y = df_train['avg_stand'].values
    X = df_train.drop(['avg_stand', 'stand', 'state', 'index'], axis=1)
    if X.isnull().values.any() == False: 

        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=8, min_samples_split=4,
                min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)

        X = polynomial_features.fit_transform(X)

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)

    else: 
        print "Found NaN values"

    rf.fit(X_train, y_train)

    p_test_data = polynomial_features.fit_transform(test_data)
    rf_pred2 = rf.predict(p_test_data)
    print rf_pred2
    test_data['avg_stand'] = rf_pred2
    final_prediction = convert_to_words(rf_pred2)
    print_full(final_prediction)
    get_position_stats(final_prediction)
    # Now we have the estimated stand_up values, we use them to create a new feature
    # in the original df
    # rf_pred3 = rf_pred2.astype(int)
    return test_data



def test_model(df_train):
    """check model accuracy, rank features"""

    y = df_train['state'].values
    X = df_train.drop(['state', 'index'], axis=1)
    
    if X.isnull().values.any() == False: 

        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=8, min_samples_split=4,
                min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
        
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)
    
    rf.fit(X_train, y_train)
    rf_prediction = rf.predict(X_test)
    rf_scores = cross_validation.cross_val_score(
    rf, X, df_train.state, cv=10, scoring='accuracy')
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    
    # report on the accuracy
    print 'Random Forest prediction accuracy this time: {}'.format(accuracy_score(y_test, rf_prediction))
    print("Random Forest general accuracy: %0.2f (+/- %0.2f)" % (rf_scores.mean(), rf_scores.std() * 2))
    # Print the feature ranking
    print("Feature importance ranking:")
    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))


def test_model_stand(df_train):
    """check model accuracy"""

    y = df_train['avg_stand'].values
    X = df_train.drop(['avg_stand', 'stand', 'state', 'index'], axis=1)
    
    if X.isnull().values.any() == False: 

        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_samples_leaf=8, min_samples_split=4,
                min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
        
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)
        
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_scores = cross_validation.cross_val_score(
    rf, X, df_train.state, cv=10, scoring='accuracy')
    
    print 'rf prediction: {}'.format(accuracy_score(y_test, rf_pred))
    print("Random Forest Accuracy: %0.2f (+/- %0.2f)" % (rf_scores.mean(), rf_scores.std() * 2))
    
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))



