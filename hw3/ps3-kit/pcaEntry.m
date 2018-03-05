%% CIS 520 Machine Learning Tyler Olivieri
% PCA

clc;clear;close all;

load('MNIST_train.mat');

% visualize one image for example
exImage = reshape(X_train(end,:), [28,28]);
figure;
imagesc(exImage');
colormap('gray');
title('example image of a 9 digit');
print -deps lastimage

% mean center dataset
xMean = mean(X_train);
xCenter = X_train - xMean;

% run pca on data
[loading,score,latent,tsquared,explained] = pca(xCenter);

% view first principal component as image
pcImage = reshape(loading(:,1), [28,28]);
figure;
imagesc(pcImage');
colormap('gray');
title('first principal component');
print -depsc 1pcimage

% find projections in 2d for digit 0 and 7
idx_0 = find(Y_train == 1);
idx_7 = find(Y_train == 8);
pc1_0 = score(idx_0, 1);
pc2_0 = score(idx_0, 2);
pc1_7 = score(idx_7, 1);
pc2_7 = score(idx_7, 2);
pc100_0 = score(idx_0, 100);
pc101_0 = score(idx_0, 101);
pc100_7 = score(idx_7, 100);
pc101_7 = score(idx_7, 101);

figure;
scatter(pc1_0, pc2_0);
hold on
scatter(pc1_7, pc2_7);
title('digits projected into first 2 principal components');
xlabel('PC 1');
ylabel('PC 2');
legend('digit 0', 'digit 7');
print -depsc 2dproj01

figure;
scatter(pc100_0, pc101_0);
hold on
scatter(pc100_7, pc101_7);
title('digits projected into 100 and 101 principal components');
xlabel('PC 100');
ylabel('PC 101');
legend('digit 0', 'digit 7');
print -depsc 2dproj100101

% observe reconstruction accuracy of % of principal components
reconAcc = cumsum(explained);
percent = [10 20 30 40 50 60 70 80 90];
for i = 1:length(percent)
    idx = find(reconAcc > percent(i));
    var(i) = min(idx);
end

figure;
plot(percent, var);
title('Reconstruction accuracy vs number of PC vectors');
xlabel('Reconstruction accuracy (%)');
ylabel('Number of PC vectors');
print -depsc percVar

% reconstruct a few images using different amounts of variances
str = "%d percent reconstruction";
var = fliplr(var);
percent = fliplr(percent);
image = [500 6000 10000 12000];
for q = 1:length(image)
    figure;
    subplot(3,4,1);
    original = reshape(X_train(image(q),:), [28,28]);
    imagesc(original');
    colormap('gray');
    title('Original image');
    for k = 1:length(var)
        reconstruct =  xMean + score(image(q),1:var(k)) * loading(:, 1:var(k))';
        subplot(3,4,k+1)
        imageRecon = reshape(reconstruct, [28,28]);
        imagesc(imageRecon');
        colormap('gray');    
        title(sprintf(str, percent(k)));
    end
    str2 = sprintf('image%d', image(q));
    print('-depsc', str2);
end
