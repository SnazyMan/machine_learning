function [W1, W2, b1, b2] =  neuralnetClassification(traindata, trainlabels, d, eta, iter)
% neural network implementation with one layer of hidden units
% traindata - mxn matrix of training data
% trainlabels - mx1 vector of training labels
% d - number of hidden units in the layer
% eta - step size (learning parameter) for gradient decent 
% iter - number of iterations in ... ****
% w - parameter matrix used for new classifications
% b - bias for new classifications

    % extract dimensionality
    [m, n] = size(traindata);
    
    % convert training labels 10 0-1 for cost function
    trainlabels = (trainlabels + 1) ./2;
    
    % read in intializations based on d
    [W1, W2, b1, b2] = init(d);
    
    % batch processing
    for i = 1:iter 
        [hiddenLayer, outputLayer] = forwardPropogation(traindata, W1, W2, b1, b2);    
        [dW1, dW2, db1, db2] = backwardsPropogation(traindata, trainlabels, W2, hiddenLayer, outputLayer);
        [W1, W2, b1, b2] = gradientDescent(W1, dW1, W2, dW2, b1, db1, b2, db2, eta);
    end
end

function [hiddenLayer, outputLayer] = forwardPropogation(traindata, W1, W2, b1, b2)
        hiddenLayer = sigmoid(traindata*W1 + b1);
        outputLayer = sigmoid(hiddenLayer*W2 + b2);
end

function [dW1, dW2, db1, db2] = backwardsPropogation(traindata, trainlabels, W2, hiddenLayer, outputLayer)
     % compute derivatives for gradient descent 
    [m,n] = size(traindata);
    [m_w] = size(W2);
    
    % dimensions (1) = mean(mx1 - mx1)
    db2 =  mean(outputLayer - trainlabels);
    % (d1x1) =  mean_m( mx1 * mxd1) [[average over all examples]]
    dW2 = (1/m) * (outputLayer - trainlabels)' * hiddenLayer;
    % (d1x1) =            1xm                           d1 x d1        d1x m                = 1xm mxd1
    db1 = (1/m) * sum((outputLayer - trainlabels) * W2'  .* (hiddenLayer .* (1 - hiddenLayer)), 1);
    %  (d1xd)            1xm d1xm mxd
    dW1 = (1/m) * ((outputLayer - trainlabels) * W2' .* hiddenLayer .* (1 - hiddenLayer))' * traindata;   
end

function [W1, W2, b1, b2] = gradientDescent(W1, dW1, W2, dW2, b1, db1, b2, db2, eta)
    % The Descent beckons !!!
    W1 = W1 - eta * dW1';
    W2 = W2 - eta * dW2';
    b1 = b1 - eta * db1;
    b2 = b2 - eta * db2;
end

function [W1, W2, b1, b2] = init(d)
    % read given initialization parameters instead of random seed
    initW1 = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Spam-Dataset/setting-files/w1_%d";
    W1p = sprintf(initW1, d);
    W1 = load(W1p);
    W1 = W1.w_1;
    initW2 = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Spam-Dataset/setting-files/w2_%d";
    W2p = sprintf(initW2, d);
    W2 = load(W2p);
    W2 = W2.w_2;
    initb1 = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Spam-Dataset/setting-files/b1_%d";
    b1p= sprintf(initb1, d);
    b1 = load(b1p);
    b1 = b1.b1;
    initb2 = "/home/snazyman/machine_learning/machine_learning/hw2/ps2-kit/Problems-1-2-3/Spam-Dataset/setting-files/b2_%d";
    b2p = sprintf(initb2, d);
    b2 = load(b2p);
    b2 = b2.b2;
end

function g = sigmoid(u)
    % Calculates sigmoid of u for logisitic regression
    g = 1.0 ./ (1.0 + exp(-u));
end

function g = sigmoidPrime(u)
    % calculates derivative of sigmoid
    g = sigmoud(u) * (1 - sigmoid(u));
end