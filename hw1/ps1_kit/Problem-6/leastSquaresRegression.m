function [w b] = leastSquaresRegression(trainingdata, traininglabels)
% leastSquaresRegression - unregularlized
%   Input - training samples - dimension m x d
%   Input - training labels - dimension m x 1
%   Output - estimated weight vector w of glm
%   Output - estimated bias b of glm

    % extract dimensionality
    [m n] = size(trainingdata);

    % augment bias column to training data
    bvector = ones(m, 1);
    X = horzcat(trainingdata, bvector);
    n = n + 1;

    % [w_hat; b_hat] =  (X_T * X)^-1 * X_T * y (normal equations)
    w = inv(X' * X) * X' * traininglabels;
    
    %extract bias
    % extract bias term from coefficient vector
    b = w(end);
    w = w(1:end - 1);    
end

