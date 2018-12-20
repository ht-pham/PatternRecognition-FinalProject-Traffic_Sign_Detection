__author__ = "Huong Pham"

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


import glob
import os
#This part is to import the dataset
'''
Some info. about the data 
- The data set contains 39,209 training images in 43 classes.
- All training images were scaled to 40x40 pixels using bi-linear interpolation and then converted to grayscale.
- Each image has 11,584 different calculated (or scaled) Haar features (5 different types, different sizes)

'''
filenames = []

def openPreprocessedImages():
    global filenames
    # The root path can be changed to your local directory where the project is downloaded
    rootpath = "/Users/huongpham/AnacondaProjects/PR-Project/precomputed-dataset/"
    dataset = dict()  # Use an empty dictionary to store all training data
    # Access 43 class ID folders to get the features of the images (in .txt)
    for c in range(0, 43):
        count = 0 # Just to keep track the number of images in each class
        classID = format(c, "05d")
        path = rootpath + classID
        os.chdir(path)

        for file in glob.glob("*.txt"):
            filenames.append(file)

            features = pd.read_csv(path+"/"+file, sep="\n", header=None)

            images = np.append(features, c)
            dataset[file] = np.transpose(images)  # Add the image's data into the dataset

            count = count + 1
            if (count == 1): # Just to print the first image of the class
                print("The first image of the %s class:" % classID)
                print(dataset[file])

        print("This "+classID + " class has "+ str(count) + " images")

    dataframe = pd.DataFrame.from_dict(dataset, orient='index')
    print(dataframe)
    return dataframe


def scaling(dataframe):
    '''
    This function is to scale the features in the dataset by removing the mean and scaling to unit variance
    :param dataframe:
    :return: dataframe with scaled value
    '''

    # Separating out the features
    features = dataframe.iloc[:,0:11584]
    scaled_features = StandardScaler().fit_transform(features)
    scaled_features_ = pd.DataFrame(scaled_features)
    dataframe.update(scaled_features_)

    return dataframe


# Define global variables
training_features = None
training_label = None
testing_features = None
testing_label = None


def split(dataframe):
    '''
    This function is to split the data set into 2 subset: 70 % for training & 30 % for testing
    :param dataframe:
    :return: training,testing
    '''
    training, testing = train_test_split(dataframe, test_size=0.30, random_state=42)

    # Separate features and class id in training set
    global training_features
    training_features = training.iloc[:,0:11584] # We have 11,584 pre-computed features
    print("%d features of all images in the training set:\n %s" % (len(training_features),training_features))
    global training_label
    training_label = training.iloc[:,11584] # one label
    print("Those images belongs to the following class IDs:\n %s" % training_label)

    # Separate features and class id in training set
    global testing_features
    testing_features = testing.iloc[:,0:11584]
    global testing_label
    testing_label = testing.iloc[:,11584]

    return training, testing


def selectBestFeatures(dataset,is_recursive=False):
    '''
    This function is for feature selection. There will be two approaches for feature selection implementation
    The first one is the univative type and the second one is the recursive approach
    :param dataset:
    :param is_recursive:
    :return: the list of the best features out of 11,584 features
    '''

    X = dataset.iloc[:,0:11584]
    y = dataset.iloc[:,11584]

    ### Univariate Feature Selection
    #   Because the task is a classification task with 43 classes, the score function f_classif is used.
    #   The score function calculates the p-score of all features. The higher the p-score of one feature is,
    #   the more important that feature is.

    def univariateFeatureSelection():
        '''
        This feature selector will pickup the best features which are 80% out of the total 11,584 features
        :return: the selected features
        '''
        selector = SelectPercentile(score_func=f_classif, percentile=20.0)
        selector.fit(X, y)  # Run score function on (X, y) and get the appropriate features' scores.

        pscores = selector.pvalues_
        print("P-Scores of the features:")
        print(pscores)
        print("Indexes of the chosen features:")
        print(selector.get_support(indices=True))
        selected_features = selector.transform(X)  # Return the mask of the best features

        return selected_features

    ### Recursive Feature Selection with Cross validation
    # The approach is K-fold cross validation
    # The data set is divided into 10 subsets, and the holdout method is repeated 10 times.
    # Each time, one of the k subsets is used as the test set and the other k-1 subsets are put together to form a training set.
    # Then the average error across all k trials is computed.
    def recursiveFeatureSelection():
        estimator = SVC(kernel='linear',gamma='auto')
        selector = RFECV(estimator, cv=KFold(n_splits=3))
        selector.fit(X, y)
        print("Cross-validation Scores: %s" % selector.grid_scores_)
        print("Optimal number of features : %d" % selector.n_features_)
        print("Selected Features: %s" % selector.support_)
        print("Feature Ranking: %s" % selector.ranking_)

        selected_features = selector.transform(X) # Return the mask of the best features

        return selected_features

    if is_recursive:
        X_reduced = recursiveFeatureSelection()
    else:
        X_reduced = univariateFeatureSelection()

    return X_reduced


def extractFeatures(train_X_reduced, test_X_reduced):
    '''
    This function is to do feature extraction
    :param train_X_reduced: the list of the best features in training set resulted from feature selection
    :param test_X_reduced: the list of the best features in testing set resulted from feature selection
    :return: A set of newly transformed features
    '''
    ### Feature extraction by PCA
    pca = PCA()
    train_components = pca.fit_transform(train_X_reduced)
    test_components = pca.transform(test_X_reduced)
    explained_variance = pca.explained_variance_ratio_
    print("Explained variance ratio: %s" % explained_variance)

    return train_components, test_components


def randomForestClassification(train_finalizedFeatures,test_finalizedFeatures):
    global training_label
    rfc = RandomForestClassifier(max_depth=10, random_state=42)
    rfc.fit(train_finalizedFeatures,training_label)
    # Predict
    predicted_label = rfc.predict(test_finalizedFeatures)

    results(predicted_label)


def knearestneighbors(train_selected_features,test_selected_features):
    global training_label
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(train_selected_features,training_label)
    #predict
    predicted = knn.predict(test_selected_features)
    results(predicted)


def results(predicted_label):
    global testing_label
    print(pd.crosstab(predicted_label, testing_label,
                      rownames=['Predicted Values'],
                      colnames=['Actual Values']))
    print("Accuracy:\n %.3f" % (accuracy_score(testing_label, predicted_label) * 100), '%')



print("Loading the dataset...")
dataset = openPreprocessedImages()
print("Scaling the dataset...")
scaled_dataset = scaling(dataset)
print("Splitting the dataset into training ")
train, test = split(scaled_dataset)

### Feature Selection
print("Performing univariate feature selection: ")
# Univariate feature selection
best_of_train = selectBestFeatures(train)
print(best_of_train.shape)
best_of_test = selectBestFeatures(test)
print(best_of_test.shape)

'''
print("Performing recursive feature selection: ")
# Recursive feature selection
best_of_train_ = selectBestFeatures(train, is_recursive=True)
best_of_test_ = selectBestFeatures(test, is_recursive=True)
'''
### make copies of train and test set for various classifiers
#PCA will be skipped
train_without_PCA1 = best_of_train
test_without_PCA1 = best_of_test
train_without_PCA2 = best_of_train
test_without_PCA2 = best_of_test
#PCA will be applied
train_with_PCA1 = best_of_train
test_with_PCA1 = best_of_test
train_with_PCA2 = best_of_train
test_with_PCA2 = best_of_test

print("Performing classification after univariate feature selection step")
print("First Approach: Random Forest Tree Classification without PCA")
randomForestClassification(train_without_PCA1, test_without_PCA1)

print("Second Approach: Random Forest Tree Classification with PCA")
train_extracted, test_extracted = extractFeatures(train_with_PCA1, test_with_PCA1)
randomForestClassification(train_extracted, test_extracted)
print("{}".format("-"*100))

print("Third Approach: KNN Classification without PCA")
knearestneighbors(train_without_PCA2,test_without_PCA2)
print("Fourth Approach: KNN Classification with PCA")
train_extracted_, test_extracted_ = extractFeatures(train_with_PCA2, test_with_PCA2)
knearestneighbors(train_extracted_,test_extracted_)

'''
print("Performing classification after recursive feature selection step")
print("First Approach: Classification without PCA")
#make copies of train and test set
train_without_PCA_ = best_of_train_
test_without_PCA_ = best_of_test_
knearestneighbors(train_without_PCA_, test_without_PCA_)

print("Second Approach: Classification with PCA")
train_extracted_, test_extracted_ = extractFeatures(best_of_train_, best_of_test_)
knearestneighbors(train_extracted_, test_extracted_)
'''









