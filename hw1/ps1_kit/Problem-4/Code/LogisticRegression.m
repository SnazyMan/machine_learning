function [w, b] = LogisticRegression(traindata, trainlabels)
    % INPUT : 
    % traindata   - m X n matrix, where m is the number of training points
    % trainlabels - m X 1 vector of training labels for the training data    
    
    % OUTPUT
    % returns learnt model: w - n x 1 weight vector, b - bias term
    
     % extract dimensonality
    [m, n] = size(traindata);
    
    % Augment an extra column of ones for bias to traindata
    bvector = ones([m, 1]);
    traindata = horzcat(traindata, bvector);
    n = n + 1;
    
    % initialize (n x 1) x 1 weight vector & bias 
    w = zeros(1, n);
    b = 0; 
    
    % replace -1 labels with 0
    I = find(trainlabels == -1);
    for k = 1:length(I)
        trainlabels(I(k)) = 0;
    end
    
    % Find minimum of error function using fminunc
    x0 = [zeros(n,1)];
    w = fminunc(@(w) costFunction(w, traindata, trainlabels), x0);
    
    % extract bias term from coefficient vector
    w = w';
    b = w(1, 58);
    w = w(:, 1:end - 1);
end 
 
 function [L] = costFunction(w, traindata, trainlabels)

    [m, n] = size(traindata);
    
    % cost function for logistic regression,
    % negative log liklihood
    for i = 1:m
        L_t{i} = trainlabels(i) * log(sigmoid(traindata(i,:) * w)) ...
            + (1 - trainlabels(i)) * log(1 - sigmoid(traindata(i,:) * w));
    end
    L = -sum([L_t{:}]);

 end
 
function g = sigmoid(u)
    % Calculates sigmoid of u for logisitic regression
    g = 1.0 ./ (1.0 + exp(-u));
end
