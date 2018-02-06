%% CIS520 Machine Learning
% Tyler Olivieri
% 1-D data
clc; clear;

% load data into workspace
load("Data-set-1/Data.mat");
load("Data-set-1/Subsets.mat");

% loop through subsets
for i = 1:length(subs)
    
    % extract current data and labels
    data = cell2mat(subs(i));
    trainingdata_part = data(:,1);
    traininglabels_part = data(:,2);
    numSamples(i) = length(traininglabels_part);
        
    % obtain glm
    [w, b] = leastSquaresRegression(trainingdata_part, traininglabels_part);
        
    % obtain y_hat for training and test
    y_hat = w*trainingdata_part + b;
    y_hatTest = w*test(:,1) + b;
    
    % measure error
    err(i) = mean_squared_error(traininglabels_part, y_hat);
    errTest(i) = mean_squared_error(test(:,2), y_hatTest);
end

figure(1);
plot(numSamples, err, numSamples, errTest, '--');
title('Least squares regression error as a function of # training examples used');
xlabel('Number of training examples');
ylabel('classification error');
legend('training', 'test');
print -depsc leastsquares_error

figure(2);
scatter(test(:,1), test(:,2));
hold on;
plot(trainingdata_part, y_hat);
title('Learned linear function imposed on actaual labels');
xlabel('instance');
ylabel('label');
print -depsc leastsquares_visual

%%
% 8-D data

clc; clear;

% load data
load('Data-set-2/Subsets.mat');
load('Data-set-2/Data.mat');
load('Data-set-2/CV_Data');

% extract current data and labels
trainingdata = train(:,1:end-1);
traininglabels = train(:,end);

data = cell2mat(subs(1));
trainingdata_part = data(:,1:end-1);
traininglabels_part = data(:,end);
numSamples = length(traininglabels_part);

testdata = test(:, 1:end-1);
testlabels = test(:, end);

% estimate glm
[w_part, b_part] = leastSquaresRegression(trainingdata_part, traininglabels_part);
[w_hat, b_hat] = leastSquaresRegression(trainingdata, traininglabels);

% compute estimates from model
yhat_partTrain = trainingdata_part * w_part + b_part;
yhat_partTest = testdata * w_part + b_part;
yhat_Train = trainingdata * w_hat + b_hat;
yhat_Test = testdata * w_hat + b_hat;

% compute error
trainErr_part = mean_squared_error(traininglabels_part, yhat_partTrain);
testErr_part = mean_squared_error(testlabels, yhat_partTest);
trainErr = mean_squared_error(traininglabels, yhat_Train);
testErr = mean_squared_error(testlabels, yhat_Test);

% five fold cross validation
lambda = [.1, 1, 10, 100, 500, 1000];
for i = 1:length(lambda)
    for k = 1:5
        % read and concatonate the folds
        idx = 1:5;
        idx(k) = [];
        data_part = vertcat(cv_data_10{idx});
        trainfold_part = data_part(:,1:end-1);
        labelfold_part = data_part(:,end);
        holdout_data = cv_data_10{k};
        holdout_datap = holdout_data(:, 1:end-1);
        holdout_labelp = holdout_data(:, end);
        
        data_all = vertcat(cv_data_all{idx});
        trainfold = data_all(:,1:end-1);
        labelfold = data_all(:,end);
        holdout = cv_data_all{k};
        holdout_dataa = holdout(:, 1:end-1);
        holdout_labela = holdout(:, end);

        % train the model from folds
        [w_foldp b_foldp] = L2LeastSquaresRegression(trainfold_part, labelfold_part, lambda(i));
        [w_folda b_folda] = L2LeastSquaresRegression(trainfold, labelfold, lambda(i));
        
        % Test on hold out set for cross validation
        yhat_foldp = holdout_datap * w_foldp + b_foldp;
        yhat_folda = holdout_dataa * w_folda + b_folda;
        
        % Test on training and test set
        yhat_traincvp = trainingdata * w_foldp + b_foldp;
        yhat_traincva = trainingdata * w_folda + b_folda;
        yhat_testcvp = testdata * w_foldp + b_foldp;
        yhat_testcva = testdata * w_folda + b_folda;
        
        % measure error
        err_foldp(k) = mean_squared_error(holdout_labelp, yhat_foldp);
        err_folda(k) = mean_squared_error(holdout_labela, yhat_folda);
        err_testp(k) = mean_squared_error(testlabels ,yhat_testcvp);
        err_testa(k) = mean_squared_error(testlabels ,yhat_testcva);
        err_trainp(k) = mean_squared_error(traininglabels, yhat_traincvp);
        err_traina(k) = mean_squared_error(traininglabels, yhat_traincva);
        
    end
    
    % average error 
    err_lambda_cvpart(i) = mean(err_foldp(:));
    err_lambda_cvall(i) = mean(err_folda(:));
    err_lambda_testpart(i) = mean(err_testp(:));
    err_lambda_testall(i) = mean(err_testa(:));
    err_lambda_trainpart(i) = mean(err_trainp(:));
    err_lambda_trainall(i) = mean(err_traina(:));
end

figure;
subplot(2,1,1)
plot(log10(lambda), err_lambda_cvpart, log10(lambda), err_lambda_testpart,log10(lambda), err_lambda_trainpart);
title('classification error on partial set as function of lambda');
xlabel('lambda');
ylabel('classification error');
legend('cv', 'test', 'training')
subplot(2,1,2)
plot(log10(lambda), err_lambda_cvall, log10(lambda), err_lambda_testall,log10(lambda), err_lambda_trainall);
title('classification error on full set as a function of lambda');
xlabel('lambda');
ylabel('classification error');
legend('cv', 'test', 'training')
print -depsc l2leastsquares


