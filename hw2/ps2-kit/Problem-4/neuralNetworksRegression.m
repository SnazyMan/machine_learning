function [W1, b1, W2, b2] = neuralNetworksRegression(traindata, trainlabels, d1, eta, iter)
% neuralnetwork implementation for regression

    % extract dimensionality
    [m, n] = size(traindata);
    
    % initialize weights
    load('Data-set-1/param.mat');
    
    for i = 1:iter    
        % forward propogate
        [f, a1] = fProp(traindata, W1, W2, b1, b2);
        
        % backwards propogation to take derivative of loss function
        [dW1, dW2, db1, db2] = bProp(f, traindata, trainlabels, a1, W2);
        
        % gradient decent to minimize loss function
        [W1, W2, b1, b2] = gradientDescent(dW1, W1, dW2, W2, db1, b1, db2, b2, eta);
    end
end

function [outputLayer, hiddenLayer] = fProp(traindata, W1, W2, b1, b2)
    % propgate data forward
    
    hiddenLayer = sigmoid(traindata*W1 + b1);
    outputLayer = hiddenLayer*W2 + b2;
end

function [dW1, dW2, db1, db2] = bProp(outputLayer, traindata, trainlabels, hiddenLayer, W2)
    % compute derivatives for gradient descent 
    [m,n] = size(traindata);
    [m_w] = size(W2);
    
    % dimensions (1) = mean(mx1 - mx1)
    db2 = 2 * mean(outputLayer - trainlabels);
    % (d1x1) =  mean_m( mx1 * mxd1) [[average over all examples]]
    dW2 = (2/m) * (outputLayer - trainlabels)' * hiddenLayer;
    % (d1x1) =            1xm                           d1 x d1        d1x m                = 1xm mxd1
    db1 = (2/m) * sum((outputLayer - trainlabels) * W2'  .* (hiddenLayer .* (1 - hiddenLayer)), 1);
    %  (d1xd)            1xm d1xm mxd
    dW1 = (2/m) * ((outputLayer - trainlabels) * W2' .* hiddenLayer .* (1 - hiddenLayer))' * traindata;   
end

function [W1, W2, b1, b2] = gradientDescent(dW1, W1, dW2, W2, db1, b1, db2, b2, eta)
    % perform gradient descent 
    W1 = W1 - eta*dW1';
    W2 = W2 - eta*dW2';
    b1 = b1 - eta*db1;
    b2 = b2 - eta*db2;
end

function g = sigmoid(u)
    % Calculates sigmoid of u for logisitic regression
    g = 1.0 ./ (1.0 + exp(-u));
end
