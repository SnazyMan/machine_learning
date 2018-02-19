%% Tyler Olivieri CIS520 Machine Learning
%

clc;clear;

% one hidden layer neural network

% read train data
path = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Spam-Dataset/train.txt";
[traindata, trainlabels] = read_spamdata(path, 250);

% read test data
path_test = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Spam-Dataset/test.txt";
[testdata, testlabels] = read_spamdata(path_test, 4351);

% Range of coefficient C for soft-margin SVM
% Selected through cross validation
path_CV = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Spam-Dataset/CrossValidation/Fold%d/cv-train.txt";
path_test_CV = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Spam-Dataset/CrossValidation/Fold%d/cv-test.txt";
d = [1 5 10 15 25 50];
for k = 1:length(d)
    
    % estimate parameter matrices with neural network
    [W1, W2, b1, b2] = neuralnetworkClassification(traindata, trainlabels, d(k), .1, 5000);
    
    % make predicitions from trained network on training and test data
    yhat_train = neuralnetClassifyEx(W1, W2, b1, b2, traindata);
    yhat_test = neuralnetClassifyEx(W1, W2, b1, b2, testdata);
    
    % compute classification error
    meanErr_train(k) = classification_error(yhat_train, trainlabels);
    meanErr_test(k) = classification_error(yhat_test, testlabels);
    
    % Cross-validation
    for i = 1:5
        % Read training folds
        trainFold = sprintf(path_CV, i);
        [traindata_CV, trainlabels_CV] = read_spamdata(trainFold, 160);

        % read holdout fold
        testFold = sprintf(path_test_CV, i);
        [testdata_CV, testlabels_CV] = read_spamdata(testFold, 40);

        % estimate parameter vector with neural network
        [W1, W2, b1, b2] = neuralnetworkClassification(traindata_CV, trainlabels_CV, d(k), .1, 5000);

        % apply learned model
        yhat_CV = neuralnetClassifyEx(W1, W2, b1, b2, testdata_CV);
                
        % compute error
        err_CV(i) = classification_error(yhat_CV, testlabels_CV);
    end
    
    % average over folds
    meanErr_CV(k) = mean(err_CV(:));
end

bestd_CV = max(d(find(meanErr_CV == min(meanErr_CV(:)))))
bestd_test = max(d(find(meanErr_test == min(meanErr_test(:)))))
bestd_train = max(d(find(meanErr_train == min(meanErr_train(:)))))

figure;
plot(d, meanErr_CV, d, meanErr_test, d, meanErr_train);
title('neural network classification error as a function of d with cross-validation');
xlabel('d');
ylabel('classification error');
legend('Crossvalidation', 'test', 'train');
%set(gca, 'Xscale', 'log')
print -depsc nnClass
