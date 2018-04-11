%% Tyler Olivieri
% CIS520 Hw5 Naive Bayes semi-supervised 

clc;clear;close all;

% load data
load('data.mat');

% learn supervised naive bayes model
[w b] = naiveBayes(labeled_train_data, train_labels);

% evaluate the model on training labels and test labels
train_hat = sign(w*labeled_train_data' + b);
test_hat = sign(w*test_data' + b);

err_train = classification_error(train_hat', train_labels);
err_test = classification_error(test_hat', test_labels);
acc_train = 1 - err_train;
acc_test = 1 - err_test;

% learn unsupervised naive bayes model
[w_un,b_un] = semiSupervisedNB(labeled_train_data,train_labels,unlabeled_data);

% evaluate the model on training labels and test labels
train_hat_un = sign(w_un*labeled_train_data' + b_un);
test_hat_un = sign(w_un*test_data' + b_un);

err_train_unlabel = classification_error(train_hat_un', train_labels);
err_test_unlabel = classification_error(test_hat_un', test_labels);
acc_train_un = 1 - err_train_unlabel;
acc_test_un = 1 - err_test_unlabel;

