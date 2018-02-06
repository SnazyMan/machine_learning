function [w b] = L2LeastSquaresRegression(trainingdata, traininglabels, lambda)
%L2LeastSquaresRegression Implement L2-Regularized Least Squares Regression
% (ridge)

  % extract dimensionality
  [m n] = size(trainingdata);

  % augment bias column to training data
  bvector = ones(m, 1);
  X = horzcat(trainingdata, bvector);
  n = n + 1;

  % Don't regularize the bias term dxd identity with
  % augmented 0 row and column R (dx1)(dx1)
  I = eye(n);
  I(n,n) = 0;
    
  w = inv(X' * X + lambda*m*I) * X' * traininglabels;
    
  % extract bias term from coefficient vector
  b = w(end);
  w = w(1:end - 1);
end

