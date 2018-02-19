%% Tyler Olivieri CIS520 HW2
% k-nn

clc;clear;

% 1 nearest neighbor
k = 1;

% read train data
path = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/train.txt";
[traindata, trainlabels] = read_data(path, 200);

% read test data
path_test = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/test.txt";
[testdata, testlabels] = read_data(path_test, 1800);

yhat_train = knn(traindata, trainlabels, traindata, k);  
trainErr = classification_error(yhat_train', trainlabels);

yhat_test = knn(traindata, trainlabels, testdata, k);    
testErr = classification_error(yhat_test', testlabels);

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

idx = knn(traindata, trainlabels, xy, 1);

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
title('1NN Decision Boundary');

% label the axes.
xlabel('x1');
ylabel('x2');

print -depsc 1nnDB

%%
% k-nn

clc;clear;

% read train data
path = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/train.txt";
[traindata, trainlabels] = read_data(path, 200);

% read test data
path_test = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/test.txt";
[testdata, testlabels] = read_data(path_test, 1800);

% select best k from cross validation
path_CV = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/CrossValidation/Fold%d/cv-train.txt";
path_test_CV = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Synthetic-Dataset/CrossValidation/Fold%d/cv-test.txt";
k_coef = [1 5 9 49 99];
for k = 1:length(k_coef)
           
    yhat_train = knn(traindata, trainlabels, traindata, k_coef(k));
    yhat_test = knn(traindata, trainlabels, testdata, k_coef(k));
    
    meanErr_test(k) = classification_error(yhat_test', testlabels);
    meanErr_train(k) = classification_error(yhat_train', trainlabels);
    
    % cross validation
    for i = 1:5
        % Read training folds
        trainFold = sprintf(path_CV, i);
        [traindata_CV, trainlabels_CV] = read_data(trainFold, 160);
        
        % read holdout fold
        testFold = sprintf(path_test_CV, i);
        [testdata_CV, testlabels_CV] = read_data(testFold, 40);
   
        yhat_CV = knn(traindata_CV, trainlabels_CV, testdata_CV, k_coef(k));
   
        % compute error
        err_CV(i) = classification_error(yhat_CV', testlabels_CV);
       
    end
    
    meanErr_CV(k) = mean(err_CV(:));
end

bestk_CV = max(k_coef(find(meanErr_CV == min(meanErr_CV(:)))))
bestk_test = max(k_coef(find(meanErr_test == min(meanErr_test(:)))))
bestk_train = max(k_coef(find(meanErr_train == min(meanErr_train(:)))))

figure;
plot(k_coef, meanErr_CV, k_coef, meanErr_test, k_coef, meanErr_train);
title('knn classification error as a function of k with cross-validation');
xlabel('k');
ylabel('classification error');
legend('Crossvalidation', 'test', 'train');
%set(gca, 'Xscale', 'log')
print -depsc knnErr

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

idx = knn(traindata, trainlabels, xy, 9);

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
title('kNN Decision Boundary');

% label the axes.
xlabel('x1');
ylabel('x2');

print -depsc knnDB

%%
% knn , higher dimensional dataset

clc;clear;

% read train data
path = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Spam-Dataset/train.txt";
[traindata, trainlabels] = read_spamdata(path, 250);

% read test data
path_test = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Spam-Dataset/test.txt";
[testdata, testlabels] = read_spamdata(path_test, 4351);

% Range of coefficient k for knn
% Selected through cross validation
path_CV = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Spam-Dataset/CrossValidation/Fold%d/cv-train.txt";
path_test_CV = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Spam-Dataset/CrossValidation/Fold%d/cv-test.txt";
k_coef = [1 5 9 49 99];
for k = 1:length(k_coef)
    
    yhat_train = knn(traindata, trainlabels, traindata, k_coef(k));
    yhat_test = knn(traindata, trainlabels, testdata, k_coef(k)); 
    
    meanErr_test(k) = classification_error(yhat_test', testlabels);
    meanErr_train(k) = classification_error(yhat_train', trainlabels);
    
    % cross-validation
    for i = 1:5
        % Read training folds
        trainFold = sprintf(path_CV, i);
        [traindata_CV, trainlabels_CV] = read_spamdata(trainFold, 200);
        
        % read holdout fold
        testFold = sprintf(path_test_CV, i);
        [testdata_CV, testlabels_CV] = read_spamdata(testFold, 50);
   
        % classify
        yhat_CV = knn(traindata_CV, trainlabels_CV, testdata_CV, k_coef(k));          
    
        % compute error
        err_CV(i) = classification_error(yhat_CV', testlabels_CV);
    end
    
    meanErr_CV(k) = mean(err_CV(:));
end

bestk_CV = max(k_coef(find(meanErr_CV == min(meanErr_CV(:)))))
bestk_test = max(k_coef(find(meanErr_test == min(meanErr_test(:)))))
bestk_train = max(k_coef(find(meanErr_train == min(meanErr_train(:)))))

figure;
plot(k_coef, meanErr_CV, k_coef, meanErr_test, k_coef, meanErr_train);
title('knn classification error as a function of k with cross-validation spam');
xlabel('k');
ylabel('classification error');
legend('Crossvalidation', 'test', 'train');
%set(gca, 'Xscale', 'log')
print -depsc knnErrSpam
