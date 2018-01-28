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
    
    % Find minimum of error function using fminunc
    x0 = [zeros(n,1)];
    options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
    w = fminunc(@(w) costFunction(w, traindata, trainlabels), x0, options);
    
    % extract bias term from coefficient vector
    w = w';
    b = w(1, 58);
    w = w(:, 1:end - 1);
end 
 
 function [L, gradient] = costFunction(w, traindata, trainlabels)

    [m, n] = size(traindata);
    
    % cost function for logistic regression,
    % negative log liklihood
    for i = 1:m
        L_t{i} = trainlabels(i) * log(sigmoid(traindata(i,:) * w)) ...
            + (1 - trainlabels(i)) * log(1 - sigmoid(traindata(i,:) * w));
    end
    L = -sum([L_t{:}]);
    
    % gradient of log liklihood
    for q = 1:n
        for z = 1:m
            g_t{z} = (sigmoid(traindata(z,:) * w) - trainlabels(z)) * traindata(m, q);
        end
        gradient(q) = sum([g_t{:}]) / m;
    end
    
 end
 
function g = sigmoid(u)
    % Calculates sigmoid of u for logisitic regression
    g = 1.0 ./ (1.0 + exp(-u));
end

% Gradient ascent to maximize liklihood w.r.t w (seems to give bad
% estimation
% gradient L(w) = sum(n=1 to N) (signma(W_T * Xn) - tn) Xn
%for i = 1:m
    %w = w + (trainlabels(m) - sigmoid(traindata(m, :) * w')) * traindata(m, :);
%end