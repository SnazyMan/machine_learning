function [prediction] = neuralnetClassifyEx(W1, W2, b1, b2, testdata)
        % make a classification given Weights and biases for one layer
        prediction = sign(forwardPropogation(testdata, W1, W2, b1, b2) - .5);
end


function [outputLayer] = forwardPropogation(traindata, W1, W2, b1, b2)
        hiddenLayer = sigmoid(traindata*W1 + b1);
        outputLayer = sigmoid(hiddenLayer*W2 + b2);
end

function g = sigmoid(u)
    % Calculates sigmoid of u for logisitic regression
    g = 1.0 ./ (1.0 + exp(-u));
end