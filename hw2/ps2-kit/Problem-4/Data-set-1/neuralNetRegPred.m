function [y] = neuralNetRegPred(W1, W2, b1, b2, testdata)
% classify new examples
    z1 = testdata*W1 + b1;
    a1 = sigmoid(z1);
    y = a1*W2 + b2;

end

function g = sigmoid(u)
    % Calculates sigmoid of u for logisitic regression
    g = 1.0 ./ (1.0 + exp(-u));
end