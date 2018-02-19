%% Tyler Olivieri CIS520 Machine Learning HW2
% Support vector machines

clc;clear;

% Linear SVM

% read train data
path = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/train.txt";
[traindata, trainlabels] = read_data(path, 200);

% read test data
path_test = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/test.txt";
[testdata, testlabels] = read_data(path_test, 1800);

% Range of coefficient C for soft-margin SVM
% Selected through cross validation
path_CV = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/CrossValidation/Fold%d/cv-train.txt";
path_test_CV = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/CrossValidation/Fold%d/cv-test.txt";
C_coef = [10^-4 10^-3 10^-2 10^-1 1 10 10^2];
for k = 1:length(C_coef)
    
    % estimate parameter vector with linear SVM 
    [alpha_t, SV_t, w_t(k, :), b_t(k, :)] = svm(traindata, trainlabels, C_coef(k), 'linear', 0);
    
    yhat_test = sign(w_t(k, :)*testdata' + b_t(k, :));
    yhat_train = sign(w_t(k, :)*traindata' + b_t(k, :));
    
    meanErr_test(k) = classification_error(yhat_test', testlabels);
    meanErr_train(k) = classification_error(yhat_train', trainlabels);
    
    % Cross-validation
    for i = 1:5
        % Read training folds
        trainFold = sprintf(path_CV, i);
        [traindata_CV, trainlabels_CV] = read_data(trainFold, 160);
        
        % read holdout fold
        testFold = sprintf(path_test_CV, i);
        [testdata_CV, testlabels_CV] = read_data(testFold, 40);
   
        % estimate parameter vector with linear SVM CV
        [alpha, SV, w(k, :), b(k, :)] = svm(traindata_CV, trainlabels_CV, C_coef(k), 'linear', 0);
    
        % apply learned model
        yhat_CV = sign(w(k, :)*testdata_CV' + b(k, :));
        
        % compute error
        err_CV(i) = classification_error(yhat_CV', testlabels_CV);
        err_test(i) = classification_error(yhat_test', testlabels);
        err_train(i) = classification_error(yhat_train', trainlabels);
        
    end
    
   meanErr_CV(k) = mean(err_CV(:));
end

bestC_CV = max(C_coef(find(meanErr_CV == min(meanErr_CV(:)))))
bestC_test = max(C_coef(find(meanErr_test == min(meanErr_test(:)))))
bestC_train = max(C_coef(find(meanErr_train == min(meanErr_train(:)))))

figure;
plot(C_coef, meanErr_CV, C_coef, meanErr_test, C_coef, meanErr_train);
title('soft SVM (linear kernel) classification error as a function of C with cross-validation');
xlabel('C');
ylabel('classification error');
legend('Crossvalidation', 'test', 'train');
set(gca, 'Xscale', 'log')
print -depsc svmlinearErr

% Decision boundary
% This code plots the decision boundary for a classifier with respect
% Directly taken from http://www.peteryu.ca/tutorials/matlab/visualize_decision_boundaries

% range of decision boundary (can be changed according to the need)
xrange = [0 25];
yrange = [0 12];

% step size for how finely you want to visualize the decision boundary (can be changed according to the need)
inc = 0.01;

% generate grid coordinates. This will be the basis of the decision
% boundary visualization.
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));

% size of the (x, y) image, which will also be the size of the
% decision boundary image that is used as the plot background.
image_size = size(x);

xy = [x(:) y(:)]; % make (x,y) pairs as a bunch of row vectors.

xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
numxypairs = length(xy); % number of (x,y) pairs

idx = sign(w_t(find(C_coef == bestC_train), :)*xy' + b_t(find(C_coef == bestC_train), :));

% reshape the idx (which contains the class label) into an image.
decisionmap = reshape(idx, image_size);

figure;

%show the image
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');

% colormap for the classes:
cmap =  [0.0 0.8 1; 1 0.8 0.8];
colormap(cmap);

% plot the class test data.
temp1 = testdata((testlabels==-1),:);
temp2 = testdata((testlabels==1),:);
plot(temp1(:,1),temp1(:,2),'bx','linewidth',0.3);
plot(temp2(:,1),temp2(:,2), 'ro','linewidth',0.3);

% include legend
legend('Class +1','Class -1')
title('SVM (Linear Kernel) Decision Boundary');

% label the axes.
xlabel('x1');
ylabel('x2');

print -depsc synthLinearDB

%%
%

clc;clear;

% polynomial kernel SVM

% read train data
path = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/train.txt";
[traindata, trainlabels] = read_data(path, 200);

% read test data
path_test = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/test.txt";
[testdata, testlabels] = read_data(path_test, 1800);

% Range of coefficient C for soft-margin SVM
% Selected through cross validation
path_CV = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/CrossValidation/Fold%d/cv-train.txt";
path_test_CV = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/CrossValidation/Fold%d/cv-test.txt";
q = [1 2 3 4 5];
C_coef = [10^-4 10^-3 10^-2 10^-1 1 10 10^2];
for z = 1:length(q);
    for k = 1:length(C_coef)
        % estimate parameter vector with linear SVM 
        [alpha_t, SV_t, w_t(z, :), b_t(z, :)] = svm(traindata, trainlabels, C_coef(k), 'polynomial', q(z));
    
        yhat_test = predictSvmQuadKernel(SV_t, alpha_t, trainlabels, traindata, testdata, b_t(z,:), q(z));
        yhat_train = predictSvmQuadKernel(SV_t, alpha_t, trainlabels, traindata, traindata, b_t(z,:), q(z));
         
        % this is a bit of a hack, but after observing best results, I want
        % to save them so I can plot
        if (q(z) == 4 &  C_coef(k) == .01)
            alpha_p = alpha_t;
            SV_p = SV_t;
            b_p = b_t(z,:);
        end
        
        meanErr_test(k) = classification_error(yhat_test', testlabels);
        meanErr_train(k) = classification_error(yhat_train', trainlabels);
        
        % cross-validation
        for i = 1:5
            % Read training folds
            trainFold = sprintf(path_CV, i);
            [traindata_CV, trainlabels_CV] = read_data(trainFold, 160);

            % read holdout fold
            testFold = sprintf(path_test_CV, i);
            [testdata_CV, testlabels_CV] = read_data(testFold, 40);

            % estimate parameter vector with linear SVM
            [alpha, SV, w(z, :), b] = svm(traindata_CV, trainlabels_CV, C_coef(k), 'polynomial', q(z));

            % apply learned model
            yhat_CV = predictSvmQuadKernel(SV, alpha, trainlabels_CV, traindata_CV, testdata_CV, b, q(z));

            % compute error
            err_CV(i) = classification_error(yhat_CV', testlabels_CV);
        end
        
        % average over folds
        meanErr_CV(k) = mean(err_CV(:));
    end
    
    % choose best C for each q
    bestC_CV(z) = max(C_coef(meanErr_CV == min(meanErr_CV(:))));
    bestC_test(z) = max(C_coef(meanErr_test == min(meanErr_test(:))));
    bestC_train(z) = max(C_coef(meanErr_train == min(meanErr_train(:))));
    
    % store error at best C for each q
    errQ_CV(z) = meanErr_CV(find(C_coef == bestC_CV(z)));
    errQ_test(z) = meanErr_test(find(C_coef == bestC_test(z)));
    errQ_train(z) = meanErr_train(find(C_coef == bestC_train(z)));
end

bestQ_CV = max(q(find(errQ_CV == min(errQ_CV(:)))))
bestQ_train = max(q(find(errQ_train == min(errQ_train(:)))))
bestQ_test = max(q(find(errQ_test == min(errQ_test(:)))))

figure;
plot(q, errQ_CV, q, errQ_test, q, errQ_train);
title('soft SVM (polynomial kernel) classification error as a function of q with cross-validation');
xlabel('q');
ylabel('classification error');
legend('Crossvalidation', 'test', 'train');
%set(gca, 'Xscale', 'log')
print -depsc svmPolynomialErr

% Decision boundary
% This code plots the decision boundary for a classifier with respect
% Directly taken from http://www.peteryu.ca/tutorials/matlab/visualize_decision_boundaries

% range of decision boundary (can be changed according to the need)
xrange = [0 25];
yrange = [0 12];

% step size for how finely you want to visualize the decision boundary (can be changed according to the need)
inc = 0.01;

% generate grid coordinates. This will be the basis of the decision
% boundary visualization.
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));

% size of the (x, y) image, which will also be the size of the
% decision boundary image that is used as the plot background.
image_size = size(x);

xy = [x(:) y(:)]; % make (x,y) pairs as a bunch of row vectors.

xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
numxypairs = length(xy); % number of (x,y) pairs

%idx = sign(w_t(find(q == bestQ_train), :)*xy' + b_t(find(q == bestQ_train), :));
idx = predictSvmQuadKernel(SV_p, alpha_p, trainlabels, traindata, xy, b_p, q(4));

% reshape the idx (which contains the class label) into an image.
decisionmap = reshape(idx, image_size);

figure;

%show the image
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');

% colormap for the classes:
cmap =  [0.0 0.8 1; 1 0.8 0.8];
colormap(cmap);

% plot the class test data.
temp1 = testdata((testlabels==-1),:);
temp2 = testdata((testlabels==1),:);
plot(temp1(:,1),temp1(:,2),'bx','linewidth',0.3);
plot(temp2(:,1),temp2(:,2), 'ro','linewidth',0.3);

% include legend
legend('Class +1','Class -1')
title('SVM (polynomial Kernel) Decision Boundary');

% label the axes.
xlabel('x1');
ylabel('x2');

print -depsc synthPolyDB

%%
%

clc;clear;

% rbf kernel SVM

% read train data
path = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/train.txt";
[traindata, trainlabels] = read_data(path, 200);

% read test data
path_test = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/test.txt";
[testdata, testlabels] = read_data(path_test, 1800);

% Range of coefficient C for soft-margin SVM
% Selected through cross validation
path_CV = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/CrossValidation/Fold%d/cv-train.txt";
path_test_CV = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/CrossValidation/Fold%d/cv-test.txt";
gamma = [10^-2 10^-1 1 10 10^2];
C_coef = [10^-3 10^-2 10^-1 1 10 10^2];
for z = 1:length(gamma);
    for k = 1:length(C_coef)
         % estimate parameter vector with rbf kernel SVM 
        [alpha_t, SV_t, w_t(z, :), b_t(z, :)] = svm(traindata, trainlabels, C_coef(k), 'rbf', gamma(z));
    
        yhat_test = predictSvmRbfKernel(SV_t, alpha_t, trainlabels, traindata, testdata, b_t(z,:), gamma(z));
        yhat_train = predictSvmRbfKernel(SV_t, alpha_t, trainlabels, traindata, traindata, b_t(z,:), gamma(z));
        
        % hack to save plotting boundary values that give best test error
        if (gamma(z) == .1 & C_coef(k) == .1)
            SV_p = SV_t;
            alpha_p = alpha_t;
            b_p = b_t(z,:);
        end
        
        meanErr_test(k) = classification_error(yhat_test', testlabels);
        meanErr_train(k) = classification_error(yhat_train', trainlabels);
        
        for i = 1:5
            % Read training folds
            trainFold = sprintf(path_CV, i);
            [traindata_CV, trainlabels_CV] = read_data(trainFold, 160);

            % read holdout fold
            testFold = sprintf(path_test_CV, i);
            [testdata_CV, testlabels_CV] = read_data(testFold, 40);

            % estimate parameter vector with linear SVM
            [alpha, SV, w, b] = svm(traindata_CV, trainlabels_CV, C_coef(k), 'rbf', gamma(z));

            % apply learned model
            yhat_CV= predictSvmRbfKernel(SV, alpha, trainlabels_CV, traindata_CV, testdata_CV, b, gamma(z));

            % compute error
            err_CV(i) = classification_error(yhat_CV', testlabels_CV);

        end
        % average over folds
        meanErr_CV(k) = mean(err_CV(:));
    end
    
    % choose best C for each gamma
    bestC_CV(z) = max(C_coef(meanErr_CV == min(meanErr_CV(:))));
    bestC_test(z) = max(C_coef(meanErr_test == min(meanErr_test(:))));
    bestC_train(z) = max(C_coef(meanErr_train == min(meanErr_train(:))));
    
    % store error at best C for each gamma
    errGamma_CV(z) = meanErr_CV(find(C_coef == bestC_CV(z)));
    errGamma_test(z) = meanErr_test(find(C_coef == bestC_test(z)));
    errGamma_train(z) = meanErr_train(find(C_coef == bestC_train(z)));
end

bestGamma_CV = max(gamma(find(errGamma_CV == min(errGamma_CV(:)))))
bestGamma_train = max(gamma(find(errGamma_train == min(errGamma_train(:)))))
bestGamma_test = max(gamma(find(errGamma_test == min(errGamma_test(:)))))

figure;
plot(gamma, errGamma_CV, gamma, errGamma_test, gamma, errGamma_train);
title('soft SVM (rbf kernel) classification error as a function of gamma with cross-validation');
xlabel('gamma');
ylabel('classification error');
legend('Crossvalidation', 'test', 'train');
set(gca, 'Xscale', 'log')
print -depsc svmRbfErr

% Decision boundary
% This code plots the decision boundary for a classifier with respect
% Directly taken from http://www.peteryu.ca/tutorials/matlab/visualize_decision_boundaries

% range of decision boundary (can be changed according to the need)
xrange = [0 25];
yrange = [0 12];

% step size for how finely you want to visualize the decision boundary (can be changed according to the need)
inc = 0.01;

% generate grid coordinates. This will be the basis of the decision
% boundary visualization.
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));

% size of the (x, y) image, which will also be the size of the
% decision boundary image that is used as the plot background.
image_size = size(x);

xy = [x(:) y(:)]; % make (x,y) pairs as a bunch of row vectors.

xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
numxypairs = length(xy); % number of (x,y) pairs

idx = predictSvmRbfKernel(SV_p, alpha_p, trainlabels, traindata, xy, b_p, .1);

% reshape the idx (which contains the class label) into an image.
decisionmap = reshape(idx, image_size);

figure;

%show the image
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');

% colormap for the classes:
cmap =  [0.0 0.8 1; 1 0.8 0.8];
colormap(cmap);

% plot the class test data.
temp1 = testdata((testlabels==-1),:);
temp2 = testdata((testlabels==1),:);
plot(temp1(:,1),temp1(:,2),'bx','linewidth',0.3);
plot(temp2(:,1),temp2(:,2), 'ro','linewidth',0.3);

% include legend
legend('Class +1','Class -1')
title('SVM (rbf Kernel) Decision Boundary');

% label the axes.
xlabel('x1');
ylabel('x2');

print -depsc sythRBFDB

%% Tyler Olivieri CIS520 Machine Learning HW2
% Support vector machines

clc;clear;

% Linear spam SVM

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
C_coef = [10^-4 10^-3 10^-2 10^-1 1 10 10^2];
for k = 1:length(C_coef)
    
    % estimate parameter vector with linear SVM 
    [alpha_t, SV_t, w_t(k, :), b_t(k, :)] = svm(traindata, trainlabels, C_coef(k), 'linear', 0);
    
    yhat_test = sign(w_t(k, :)*testdata' + b_t(k, :));
    yhat_train = sign(w_t(k, :)*traindata' + b_t(k, :));
    
    meanErr_test(k) = classification_error(yhat_test', testlabels);
    meanErr_train(k) = classification_error(yhat_train', trainlabels);
    
    % Cross-validation
    for i = 1:5
        % Read training folds
        trainFold = sprintf(path_CV, i);
        [traindata_CV, trainlabels_CV] = read_spamdata(trainFold, 200);
        
        % read holdout fold
        testFold = sprintf(path_test_CV, i);
        [testdata_CV, testlabels_CV] = read_spamdata(testFold, 50);
   
        % estimate parameter vector with linear SVM
        [alpha, SV, w, b] = svm(traindata_CV, trainlabels_CV, C_coef(k), 'linear', 0);
    
        % apply learned model
        yhat_CV = sign(w*testdata_CV' + b);
        
        % compute error
        err_CV(i) = classification_error(yhat_CV', testlabels_CV);
        
    end
    
   meanErr_CV(k) = mean(err_CV(:));
end

bestC_CV = max(C_coef(find(meanErr_CV == min(meanErr_CV(:)))))
bestC_test = max(C_coef(find(meanErr_test == min(meanErr_test(:)))))
bestC_train = max(C_coef(find(meanErr_train == min(meanErr_train(:)))))

figure;
plot(C_coef, meanErr_CV, C_coef, meanErr_test, C_coef, meanErr_train);
title('soft SVM (linear kernel) classification error as a function of C with cross-validation');
xlabel('C');
ylabel('classification error');
legend('Crossvalidation', 'test', 'train');
set(gca, 'Xscale', 'log')
print -depsc svmlinearErr_spam


%%
%

clc;clear;

% polynomial kernel SVM

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
q = [1 2 3 4 5];
C_coef = [10^-4 10^-3 10^-2 10^-1 1 10 10^2];
for z = 1:length(q);
    for k = 1:length(C_coef)   
         % estimate parameter vector with polynomial SVM 
        [alpha_t, SV_t, w_t(z, :), b_t(z, :)] = svm(traindata, trainlabels, C_coef(k), 'polynomial', q(z));
    
        yhat_test = predictSvmQuadKernel(SV_t, alpha_t, trainlabels, traindata, testdata, b_t(z,:), q(z));
        yhat_train = predictSvmQuadKernel(SV_t, alpha_t, trainlabels, traindata, traindata, b_t(z,:), q(z));
        
        meanErr_test(k) = classification_error(yhat_test', testlabels);
        meanErr_train(k) = classification_error(yhat_train', trainlabels);
        
        % cross-validation
        for i = 1:5
            % Read training folds
            trainFold = sprintf(path_CV, i);
            [traindata_CV, trainlabels_CV] = read_spamdata(trainFold, 160);

            % read holdout fold
            testFold = sprintf(path_test_CV, i);
            [testdata_CV, testlabels_CV] = read_spamdata(testFold, 40);

            % estimate parameter vector with linear SVM
            [alpha, SV, w, b] = svm(traindata_CV, trainlabels_CV, C_coef(k), 'polynomial', q(z));

            % apply learned model
            yhat_CV = predictSvmQuadKernel(SV, alpha, trainlabels_CV, traindata_CV, testdata_CV, b, q(z));

            % compute error
            err_CV(i) = classification_error(yhat_CV', testlabels_CV);
        end
        
        % average over folds
        meanErr_CV(k) = mean(err_CV(:));
    end
    
    % choose best C for each q
    bestC_CV(z) = max(C_coef(meanErr_CV == min(meanErr_CV(:))));
    bestC_test(z) = max(C_coef(meanErr_test == min(meanErr_test(:))));
    bestC_train(z) = max(C_coef(meanErr_train == min(meanErr_train(:))));
    
    % store error at best C for each q
    errQ_CV(z) = meanErr_CV(find(C_coef == bestC_CV(z)));
    errQ_test(z) = meanErr_test(find(C_coef == bestC_test(z)));
    errQ_train(z) = meanErr_train(find(C_coef == bestC_train(z)));
end

bestQ_CV = max(q(find(errQ_CV == min(errQ_CV(:)))))
bestQ_train = max(q(find(errQ_train == min(errQ_train(:)))))
bestQ_test = max(q(find(errQ_test == min(errQ_test(:)))))

figure;
plot(q, errQ_CV, q, errQ_test, q, errQ_train);
title('soft SVM (polynomial kernel) classification error as a function of q with cross-validation');
xlabel('q');
ylabel('classification error');
legend('Crossvalidation', 'test', 'train');
%set(gca, 'Xscale', 'log')
print -depsc svmPolynomialErr_spam

%%

clc;clear;

% rbf kernel SVM

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
gamma = [10^-2 10^-1 1 10 10^2];
C_coef = [10^-3 10^-2 10^-1 1 10 10^2];
for z = 1:length(gamma);
    for k = 1:length(C_coef)
        % estimate parameter vector with rbf kernel SVM 
        [alpha_t, SV_t, w_t(z, :), b_t(z, :)] = svm(traindata, trainlabels, C_coef(k), 'rbf', gamma(z));
    
        yhat_test = predictSvmRbfKernel(SV_t, alpha_t, trainlabels, traindata, testdata, b_t(z,:), gamma(z));
        yhat_train = predictSvmRbfKernel(SV_t, alpha_t, trainlabels, traindata, traindata, b_t(z,:), gamma(z));
        
        meanErr_test(k) = classification_error(yhat_test', testlabels);
        meanErr_train(k) = classification_error(yhat_train', trainlabels);
        
        % cross validation
        for i = 1:5
            % Read training folds
            trainFold = sprintf(path_CV, i);
            [traindata_CV, trainlabels_CV] = read_spamdata(trainFold, 160);

            % read holdout fold
            testFold = sprintf(path_test_CV, i);
            [testdata_CV, testlabels_CV] = read_spamdata(testFold, 40);

            % estimate parameter vector with linear SVM
            [alpha, SV, w, b] = svm(traindata_CV, trainlabels_CV, C_coef(k), 'rbf', gamma(z));

            % apply learned model
            yhat_CV = predictSvmRbfKernel(SV, alpha, trainlabels_CV, traindata_CV, testdata_CV, b, gamma(z));
         
            % compute error
            err_CV(i) = classification_error(yhat_CV', testlabels_CV);
        end
        
        % average over folds
        meanErr_CV(k) = mean(err_CV(:));
    end
    
    % choose best C for each gamma
    bestC_CV(z) = max(C_coef(meanErr_CV == min(meanErr_CV(:))));
    bestC_test(z) = max(C_coef(meanErr_test == min(meanErr_test(:))));
    bestC_train(z) = max(C_coef(meanErr_train == min(meanErr_train(:))));
    
    % store error at best C for each gamma
    errGamma_CV(z) = meanErr_CV(find(C_coef == bestC_CV(z)));
    errGamma_test(z) = meanErr_test(find(C_coef == bestC_test(z)));
    errGamma_train(z) = meanErr_train(find(C_coef == bestC_train(z)));
end

bestGamma_CV = max(gamma(find(errGamma_CV == min(errGamma_CV(:)))))
bestGamma_train = max(gamma(find(errGamma_train == min(errGamma_train(:)))))
bestGamma_test = max(gamma(find(errGamma_test == min(errGamma_test(:)))))

figure;
plot(gamma, errGamma_CV, gamma, errGamma_test, gamma, errGamma_train);
title('soft SVM (rbf kernel) classification error as a function of gamma with cross-validation');
xlabel('gamma');
ylabel('classification error');
legend('Crossvalidation', 'test', 'train');
set(gca, 'Xscale', 'log')
print -depsc svmRbfErr_spam
