%% Tyler Olivieri CIS520 HW2
%

% Neural networks for regression
clc;clear;

% read data
load("Data.mat");
load("Subsets.mat");

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
load("Data.mat");
load("Subsets.mat");

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
    [W1, b1, W2, b2] = neuralNetworksRegressionRELU(traindata, trainlabels, d1, eta, iter); 

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

 