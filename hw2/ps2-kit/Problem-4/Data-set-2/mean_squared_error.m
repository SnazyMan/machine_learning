function [res] = mean_squared_error(y, y_hat)
    res = sum((y - y_hat).^2) / size(y,1);
end