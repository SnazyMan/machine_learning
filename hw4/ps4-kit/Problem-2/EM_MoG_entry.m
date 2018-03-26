%% Tyler Olivieri CIS520 
% Mixture of Gaussians

clc;clear;close all;

K = 3;
load('X_test.mat');

subsets = [.1 .2 .3 .4 .5 .6 .7 .8 .9 1];
for i = 1:(length(subsets)-1)
    % load data
    str = sprintf("./TrainSubsets/X_%.1f.mat", subsets(i));
    load(str);
    str = sprintf("./MeanInitialization/Part_a/mu_%.1f.mat", subsets(i));
    load(str);
    
    % learn mixture of gaussian model
    [pi,mu_out,cov] = EM_MoG(X,K,mu);
    
    % compute normalized log liklihood of model and train data
    train_llh(i) = compute_nllh(X, K, mu_out, cov, pi);
    test_llh(i)= compute_nllh(X_test, K, mu_out, cov, pi); 
    
    sample_plot(X_test,mu_out,cov,subsets(i));    
end

% do last edge case subset
load("./TrainSubsets/X_1.mat");
load("./MeanInitialization/Part_a/mu_1.mat");
[pi,mu_out,cov] = EM_MoG(X,K,mu);
train_llh(i+1) = compute_nllh(X, K, mu_out, cov, pi);
test_llh(i+1)= compute_nllh(X_test, K, mu_out, cov, pi);

sample_plot(X_test,mu_out,cov,subsets(i+1));    

% plot learning curve
figure;
plot(subsets,test_llh,subsets,train_llh)
title('Learning curve');
xlabel('Percent of training data used');
ylabel('normalized log liklihood');
legend('test','training');
print -depsc learningcurve

%%

clc;clear;close all;

% read train data
load("./X.mat")
% read test data
load("./X_test.mat")

% Range of K gaussians
% Selected through cross validation
path_CV = "./CrossValidation/Fold%d/cv-train.mat";
path_test_CV = "./CrossValidation/Fold%d/cv-test.mat";
path_mu = "./MeanInitialization/Part_b/mu_k_%d.mat";
K = [1 2 3 4 5];
for i = 1:length(K)
    
    % load mu
    str = sprintf(path_mu,i);
    load(str);   
    
    % learn mixture of gaussians
    [pi,mu_out,cov] = EM_MoG(X_full, K(i), mu);
    
    % compute train and test liklihood
    train_llh(i) = compute_nllh(X_full, K(i), mu_out, cov, pi);
    test_llh(i)= compute_nllh(X_test, K(i), mu_out, cov, pi);
    
    % Cross-validation
    for z = 1:5
        % Read training folds
        trainFold = sprintf(path_CV, z);
        load(trainFold);
        
        % read holdout fold
        testFold = sprintf(path_test_CV, z);
        load(testFold);
        
        % learn mixture of gaussians
        [pi,mu_out,cov] = EM_MoG(X_train, K(i), mu);
        
        % compute train and test liklihood
        CV_llh(z) = compute_nllh(X_test, K(i), mu_out, cov, pi);
    end
    
    % average over folds
    meanErr_CV(i) = mean(CV_llh(:));
end

bestK_CV = max(K(find(meanErr_CV == max(meanErr_CV(:)))))

figure;
plot(K, meanErr_CV, K, test_llh, K, train_llh);
title('MoG liklihood as a function of K with cross-validation');
xlabel('K');
ylabel('classification error');
legend('Crossvalidation', 'test', 'train');
%set(gca, 'Xscale', 'log')
print -depsc MoGCV

