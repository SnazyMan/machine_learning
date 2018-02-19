function [W1, b1, W2, b2] = neuralNetworksRegressionRELU(traindata, trainlabels, d1, eta, iter, data)
% neural network training method that uses ReLU activation functions

    % extract dimensionality
    [m, n] = size(traindata);
    
    if (data == 0)
        load('Data-set-1/param.mat');
    else
        % initialize weights
        path = sprintf("Data-set-2/param_%d.mat", d1);
        load(path);
    end
    
    for i = 1:iter    
        % forward propogate
        [f, a1] = fProp(traindata, W1, W2, b1, b2);
        
        % backwards propogation to take derivative of loss function
        [dW1, dW2, db1, db2] = bProp(f, traindata, trainlabels, a1, W2, dReLU(traindata*W1 + b1));
        
        % gradient decent to minimize loss function
        [W1, W2, b1, b2] = gradientDescent(dW1, W1, dW2, W2, db1, b1, db2, b2, eta);
    end
end

function [f, a1] = fProp(traindata, W1, W2, b1, b2)
    % propgate data forward
    z1 = traindata*W1 + b1;
    a1 = ReLU(z1);
    f = a1*W2 + b2;
end

function [dW1, dW2, db1, db2] = bProp(outputLayer, traindata, trainlabels, hiddenLayer, W2, dReLU)

    % compute derivatives for gradient descent 
    [m,n] = size(traindata);
    [m_w] = size(W2);
    
    % dimensions (1) = mean(mx1 - mx1)
    db2 = 2 * mean(outputLayer - trainlabels);
    % (d1x1) =  mean_m( mx1 * mxd1) [[average over all examples]]
    dW2 = (2/m) * (outputLayer - trainlabels)' * hiddenLayer;
    % (d1x1) =            1xm                           d1 x d1        d1x m                = 1xm mxd1
    db1 = (2/m) * sum((outputLayer - trainlabels) * W2' .* dReLU, 1);
    %  (d1xd)            1xm d1xm mxd
    dW1 = (2/m) * ((outputLayer - trainlabels) * W2' .* dReLU)' * traindata; 
    
end

function [W1, W2, b1, b2] = gradientDescent(dW1, W1, dW2, W2, db1, b1, db2, b2, eta)
    % perform gradient descent 
    W1 = W1 - eta*dW1';
    W2 = W2 - eta*dW2';
    b1 = b1 - eta*db1;
    b2 = b2 - eta*db2;
end

function g = ReLU(u)
    % Calculates ReLU of u
    g = max(0, u);
end

function g = dReLU(u)
    % Calculates subderivative of ReLU
    u(u<0) = 0;
    u(u>0) = 1;
    g = u;
end



