%% Tyler Olivieri CIS520 HW2
%

% Neural networks for regression
clc;clear;

% read data
load("Data-set-1/Data.mat");
load("Data-set-1/Subsets.mat");

eta = .1;
iter = 20000;
d1 = 15;

% loop over subsets to create a learning curve 
for i = 1:length(subs)
    % extract current data and labels
    data = cell2mat(subs(i));
    traindata = data(:,1);
    trainlabels = data(:,2);
    numSamples(i) = length(trainlabels);
    
    % train network
    [W1, b1, W2, b2] = neuralNetworksRegression(traindata, trainlabels, d1, eta, iter); 
    
    % make predictions
    yhat_train = neuralNetRegPred(W1, W2, b1, b2, traindata);
    yhat_test = neuralNetRegPred(W1, W2, b1, b2, test(:,1));
    
    % compute error
    err_train(i) = mean_squared_error(trainlabels, yhat_train);
    err_test(i) = mean_squared_error(test(:,2), yhat_test);
end

figure;
plot(numSamples, err_train, numSamples, err_test, '--');
title('neural regression regression error as a function of # training examples used');
xlabel('Number of training examples');
ylabel('classification error');
legend('training', 'test');
print -depsc neuralnet_error

figure;
scatter(test(:,1), test(:,2));
hold on;
stem(traindata, yhat_train);
title('Learned linear function imposed on actaual labels');
xlabel('instance');
ylabel('label');
print -depsc neuralnet_visual

%%
%

% ReLU units

% Neural networks for regression
clc;clear;

% read data
load("Data-set-1/Data.mat");
load("Data-set-1/Subsets.mat");

eta = .1;
iter = 20000;
d1 = 15;

% loop over subsets to create a learning curve 
for i = 1:length(subs)
    % extract current data and labels
    data = cell2mat(subs(i));
    traindata = data(:,1);
    trainlabels = data(:,2);
    numSamples(i) = length(trainlabels);
    
    % train network
    [W1, b1, W2, b2] = neuralNetworksRegressionRELU(traindata, trainlabels, d1, eta, iter, 0); 

    % make predictions
    yhat_train = neuralNetRegPredRELU(W1, W2, b1, b2, train(:,1));
    yhat_test = neuralNetRegPredRELU(W1, W2, b1, b2, test(:,1));
    
    % compute error
    err_train(i) = mean_squared_error(train(:,2), yhat_train);
    err_test(i) = mean_squared_error(test(:,2), yhat_test);
end

figure;
plot(numSamples, err_train, numSamples, err_test, '--');
title('neural regression regression error as a function of # training examples used RELU activation');
xlabel('Number of training examples');
ylabel('classification error');
legend('training', 'test');
print -depsc neuralnet_errorRELU

figure;
scatter(test(:,1), test(:,2));
hold on;
stem(traindata, yhat_train);
title('Learned linear function imposed on actaual labels RELU activation');
xlabel('instance');
ylabel('label');
print -depsc neuralnet_visualRELU

%%
% Predict concrete compressive strength

clc;clear;

% read data
load("Data-set-2/Data.mat");
load("Data-set-2/Subsets.mat");
load("Data-set-2/CV_Data.mat");
traindata = train(:,1:end-1);
trainlabels = train(:,end);
testdata = test(:,1:end-1);
testlabels = test(:, end);

eta = 3.5e-6;
iter = 20000;

d1 = [7 10 15 17 20];
% five fold cross validation
for i = 1:length(d1)
    
    % train network
    [W1, b1, W2, b2] = neuralNetworksRegressionRELU(traindata, trainlabels, d1(i), eta, iter, 1); 

    % make predictions
    yhat_train = neuralNetRegPredRELU(W1, W2, b1, b2, traindata);
    yhat_test = neuralNetRegPredRELU(W1, W2, b1, b2, testdata);
    
    % compute error
    err_train(i) = mean_squared_error(trainlabels, yhat_train);
    err_test(i) = mean_squared_error(testlabels, yhat_test);
    
    for k = 1:5
        % read and concatonate the folds
        idx = 1:5;
        idx(k) = [];
        
        data = vertcat(cv_data_all{idx});
        traindata_CV = data(:,1:end-1);
        trainlabels_CV = data(:,end);
        holdout = cv_data_all{k};
        testdata_CV = holdout(:, 1:end-1);
        testlabels_CV = holdout(:, end);

        % train network
        [W1, b1, W2, b2] = neuralNetworksRegressionRELU(traindata_CV, trainlabels_CV, d1(i), eta, iter, 1); 
        
        % make predictions
        yhat_CV = neuralNetRegPredRELU(W1, W2, b1, b2, testdata_CV);
        
        % measure error
        err_CV(k) = mean_squared_error(testlabels_CV, yhat_CV);
    end
    
    % average error 
    meanErr_CV(i) = mean(err_CV(:));
end

bestd_CV = max(d1(find(meanErr_CV == min(meanErr_CV(:)))))
bestd_test = max(d1(find(err_test == min(err_test(:)))))
bestd_train = max(d1(find(err_train == min(err_train(:)))))

figure(1);
plot(d1, meanErr_CV, d1, err_test, d1, err_train);
title('neural network classification error as a function of d with cross-validation');
xlabel('d1');
ylabel('classification error');
legend('Crossvalidation', 'test', 'train');
%set(gca, 'Xscale', 'log')
print -depsc nnClassConcrete


