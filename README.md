## CSC551 - Final Project 
# Application of Pattern Recognition on Traffic Sign Detection

### Project Description:
* The project's purpose is to learn and understand the whole pattern recognition process from the data cleansing steps (data preprocessing, feature selection and feature extraction) to the classification steps (training data and testing the model)
* The techniques used: Data Standardization for preprocessing, both univariate and recursive feature selection techniques for comparison, principal components analysis for feature extractions, and two classifiers also for comparisons
* Programming Language: Python with the supporting libraries: sklearn, pandas, and numpy
### Related Work:
In 2011, the German Traffic Signs Benchmark team conducted a project about traffic road sign detection in German. The project was intensive and impressive because 
* The dataset has a collection of more than 50,000 images of 43 different road signs and all of the images were pre-computed in different sizes for 12 features. 
* Each image has an overall feature vector of 11,584 features.
### Data collection & Preprocessing:
- The dataset used in this project is the training dataset of the mentioned related work and it is splitted into two subsets for training and testing purposes. 
- The dataset is a collection of 2251 .txt files, each of which contains 11,584 pre-computed Haar-like features for one image. If you are interested, you can download [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Pre-calculated_feature).
- All of the files are categorized into 43 different classes.
#### Data preprocessing:
The features are converted into the standard by removing the mean and scaling to unit variance
### Feature Selection:
Overall, there are two techniques of feature selection including:
1. Univariate Feature Selection:
- This is the technique that selects the best features only one time based on our preference.
- The desired number of best features is chosen either by a certain number M out of N features or by m percent of N features. The second option is implemented in the univariate feature selection step of this project due to the huge number of the features.
- The criteria to choose the best features bases on the ratio of the variance between the mean samples and the variance within the sample. The larger the ratio is, the better the classification will be. The f-contribution test is used to compute the ratio for this multi-class classification project.
2. Recursive Feature Selection:
- This is the technique that selects features recursively by considering smaller and smaller sets of features based on the assigned weights of the features computed from an estimator. The approach will initially work on the whole set to choose the best features and leave out the least important features, and then repeat the computation process on the set of the least important one until the desired number of features is reached. 
- The estimator used for this project is k-fold cross-validation.

**Note:** This project requires a CPU with huge memory or even a GPU in order to run the recursive technique because it has to compute the weights of 11,584 features per image and the dataset has 2251 images.
### Feature Extraction:
- Because the original number of features is enormously huge, it still needs reducing after feature selection. Therefore, I also performed Principal Components Analysis (PCA) to transform and reduce the dimension of the feature vector in a hope that it will eliminate more unnecessary features and benefit the classification process. 
- Another reason is that I wanted to make comparison between classification with applied PCA and classification without applied PCA. 
- The desired number of components set by default is the minimum number between the number of samples and the number of features subtracted by 1, which is 2250 components. 
### Classification Methods:
There are several classifiers for training and testing the data. I chose two methods to implement. 
1.  Random forest tree classification 
2.  k-nearest neighbor classification (The depth of the random forest tree is as default 2. The number of neighbor is 10.)
### Results:
The below table shows the accuracy rate (in percentage) of the random forest tree classification (RFC) with and without applying PCA for the dataset with various sizes of selected features:

SelectPercentile(percentile=?) (%) | Second Header | First Header 
---------------------------------- | :-------------: | :-------------:
10.0 | 81.481 | 23.259 
15.0 | 64.889 | 61.778
20.0 | 88.296 | 46.667
25.0 | 82.519 | 40.000
30.0 | 65.778 | 31.704
50.0 | 91.852 | 49.481
60.0 | 60.148 | 44.000
80.0 | 82.074 | 35.111
90.0 | 69.926 | 31.852

The K-nearest neighbor approaches I tested its accuracy with three different values of k. The first value is k=3, the accuracy rate is 87.852 %. The second value is k=5, the accuracy rate is 89.185%. The third value is k = 10, the accuracy rate is 89.185%.
### Reference:
J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In _Proceedings of the IEEE International Joint Conference on Neural Networks_, pages 1453–1460. 2011. 
