Semi-Supervised Naive Bayes via EM

You can find the data for this problem in the file data.mat

This file contains the following variables:
* labeled_train_data -> 100x100 matrix containing 100 labeled samples for training
* train_labels       -> 100x1 matrix containing the labels for the training data
* test_data          -> 1000x100 matrix containing 1000 test samples
* test_labels        -> 1000x1 matrix containing the labels for the test data
* unlabeled_data     -> 900x100 sparse matrix containing 900 unlabeled documents

In all cases, rows represent instances and columns represent features.