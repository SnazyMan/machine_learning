% %% CIS 520 Machine Learning 
% % Tyler Olivieri  
% % Logistic Regression
% 
% clc; clear;
% 
% err_idx = 0;
% 
% % test spam email
% % label of spam (+1) or not spam (-1)
% % C(i) is Xi where each Xi has 57 features and i = 58 is the label
% testFd = fopen("/home/snazyman/machine_learning/machine_learning/hw1/ps1_kit/Problem-4/Spambase/test.txt");
% testSamples = 4351;
% for z = 1:testSamples
%     C(z) = textscan(testFd, '%f', 58);
% end
% fclose(testFd);
% 
% % Seperate feature vectors and labels from cell array 
% % label goes into m x 1 vector
% % training data goes into m x n matrix
% % m is number of feature vectors
% % n is the dimension 
% % extract from cell array, transpose to satisfy above dimensions
% testFile = cell2mat(C);
% testFile = testFile';
% 
% % extract labels and remove labels from traindata matrix
% testlabels = testFile(:, end);
% testdata = testFile(:, 1:end - 1);
% 
% % learn classifier from different subsets of training data
% % loop through training files
% path = "/home/snazyman/machine_learning/machine_learning/hw1/ps1_kit/Problem-4/Spambase/TrainSubsets/train-%d%s";
% ext = "%.txt";
% for q = 10:10:100
%     % read in training subset
%     trainSubset = sprintf(path, q, ext);
%     fileID = fopen(trainSubset);
% 
%     % keep track of numSamples in classification for plotting
%     err_idx = err_idx + 1;
%     numSamples(err_idx) = 250 * (q / 100);
%     for i = 1:numSamples(err_idx)
%         cFile(i) = textscan(fileID, '%f', 58);
%     end
%     fclose(fileID);
% 
%     numFile = cell2mat(cFile);
%     numFile = numFile';
% 
%     trainlabels = numFile(:, end);
%     traindata = numFile(:, 1:end - 1);
% 
%     % Find parameter vector w, bias b using logistic regression
%     [w_hat, b_hat] = LogisticRegression(traindata, trainlabels);
%         
%     % use plug in classifier for logisitc regression with w_hat & b_hat
%     % derived from bayes optimal classifier    
%     for iter = 1:numSamples(err_idx)
%         y_hat(iter) = sign(traindata(iter, :) * w_hat' + b_hat);
%     end
%     
%     % calculate classification error on training data
%     err(err_idx) = classification_error(y_hat(:), trainlabels(:));
% 
%     % measure error on test set    
%     for testIter = 1:testSamples
%         y_hatTest(testIter) = sign(testdata(testIter, :) * w_hat' + b_hat);
%     end
%     
%     errTest(err_idx) = classification_error(y_hatTest(:), testlabels(:));
% end
% 
% % classification error on test set vs numSamples
% figure(1);
% plot(numSamples, err, numSamples, errTest, '--')
% title('Logistic regression classification error as a function of # training examples used');
% xlabel('Number of training examples');
% ylabel('classification error');
% legend('training','test')
% 
% % used for sanity check
% I = find(trainlabels == -1);
% for k = 1:length(I)
%     trainlabels(I(k)) = 0;
% end
% testW = glmfit(traindata, trainlabels, 'binomial', 'link', 'logit');
% 
% %%
% % Logistic Regression with L2 regularizer
% 
% clc; clear;
% 
% err_idx = 0;
% 
% % open/read training set and test set
% % test spam email
% % label of spam (+1) or not spam (-1)
% % C(i) is Xi where each Xi has 57 features and i = 58 is the label
% testFd = fopen("/home/snazyman/machine_learning/machine_learning/hw1/ps1_kit/Problem-4/Spambase/test.txt");
% testSamples = 4351;
% for z = 1:testSamples
%     C(z) = textscan(testFd, '%f', 58);
% end
% fclose(testFd);
% 
% % Seperate feature vectors and labels from cell array 
% % label goes into m x 1 vector
% % training data goes into m x n matrix
% % m is number of feature vectors
% % n is the dimension 
% % extract from cell array, transpose to satisfy above dimensions
% testFile = cell2mat(C);
% testFile = testFile';
% 
% % extract labels and remove labels from traindata matrix
% testlabels = testFile(:, end);
% testdata = testFile(:, 1:end - 1);
% 
% fileID = fopen("/home/snazyman/machine_learning/machine_learning/hw1/ps1_kit/Problem-4/Spambase/train.txt");
% 
% % keep track of numSamples in classification for plotting
% err_idx = err_idx + 1;
% numSamples = 250;
% for i = 1:numSamples
%     cFile(i) = textscan(fileID, '%f', 58);
% end
% fclose(fileID);
% 
% numFile = cell2mat(cFile);
% numFile = numFile';
% 
% trainlabels = numFile(:, end);
% traindata = numFile(:, 1:end - 1);
% 
% % estimate linear model with a set of lamba
% lambda = [10^-7 10^-6 10^-5 10^-4 10^-3 10^-2 10^-1 1];
% for i = 1:length(lambda)
%     % classify with estimated model
%     % Find parameter vector w, bias b using L2 logistic regression
%     [w_hat, b_hat] = LogisticRegressionL2(traindata, trainlabels, lambda(i));
%         
%     % use plug in classifier for logisitc regression with w_hat & b_hat
%     % derived from bayes optimal classifier    
%     for iter = 1:numSamples
%         y_hat(iter) = sign(traindata(iter, :) * w_hat' + b_hat);
%     end
%     
%     % calculate classification error on training data
%     err(i) = classification_error(y_hat(:), trainlabels(:));
% 
%     % measure error on test set    
%     for testIter = 1:testSamples
%         y_hatTest(testIter) = sign(testdata(testIter, :) * w_hat' + b_hat);
%     end
%     
%     errTest(i) = classification_error(y_hatTest(:), testlabels(:));
% end
% 
% % classification error on test set vs numSamples
% figure(2);
% plot(lambda, err, lambda, errTest, '--')
% title('L2-Logistic Regression classification error as a function of lambda');
% xlabel('lambda');
% ylabel('classification error');
% legend('training','test')

%%
% Logistic Regression with L2 regularizer 5-fold cross validation

clc; clear;

% read in the 5 folds
% open/read training set and test set
% test spam email
% label of spam (+1) or not spam (-1)
% C(i) is Xi where each Xi has 57 features and i = 58 is the label
path = "/home/snazyman/machine_learning/machine_learning/hw1/ps1_kit/Problem-4/Spambase/CrossValidation/Fold%d/cv-train.txt";
for i = 1:5
    trainFold = sprintf(path, i);
    trainFd = fopen(trainFold);
    numSamples = 200;
    for z = 1:numSamples
        C(z) = textscan(trainFd, '%f', 58);
    end
    fclose(trainFd);

    % Seperate feature vectors and labels from cell array 
    % label goes into m x 1 vector
    % training data goes into m x n matrix
    % m is number of feature vectors
    % n is the dimension 
    % extract from cell array, transpose to satisfy above dimensions
    trainFile = cell2mat(C);
    trainFile = trainFile';

    % extract labels and remove labels from traindata matrix
    trainlabels = trainFile(:, end);
    traindata = trainFile(:, 1:end - 1);
    
    trainlabels_F(:,:,i) = trainlabels;
    traindata_F(:,:,i) = traindata;
end

% permute the 5 folds 5 times, estimate linear model over different
% regularization
i