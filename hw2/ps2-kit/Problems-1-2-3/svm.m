function [alpha, SV, w, b] = svm(traindata, trainlabels, C, ...
                                kernel_type, kernel_param)
%svm support vector machine implementation
%   Input - trainingdata
%   Input - traininglabels
%   Input - C - soft-margin SVM, controls tradeoff between increasing
%           margin and reducing errors
%   Input - kernel_type - 'linear', 'polynomial', or 'rbf'
%   Input - kernel_param - real number: parameter q in polynomial kernel
%           or parameter gamma in rbf kernel
%   Output - kernel type and parameters
%   Output - sv - support vectors
%   Output - alpha - coefficients corresponding to support vectors
%   Output - b - bias term

    % Determine if valid kernel type is passed to func
    if (strcmp(kernel_type, 'linear'))
        K = traindata * traindata';
    elseif (strcmp(kernel_type, 'polynomial'))
        K = (traindata * traindata' + 1).^kernel_param;
    elseif (strcmp(kernel_type, 'rbf'))
       K = rbf_kernel(traindata, kernel_param);
    else
        error("Not a valid kernel type");
    end
        
    % extract dimensionality
    [m,n] = size(traindata);
    
    % alpha coefficients - negate to make convex
    f = repmat(-1,m,1);
    
    % alpha^2 coefficient matrix - negated to make convex
    % m by m matrix
    % at matrix postion (m,n), y_m * y_n *(x_mT*x_n)
    H = (trainlabels * trainlabels').*K;

    % constraint 0 <= alpha <= C
    lb = zeros(m,1);
    ub = repmat(C,m,1);
    
    % constraint <alpha_m, traininglabels_m> = 0
    Aeq = trainlabels';
    beq = 0;
    
    % solve the QP dual problem (lagrange for alpha) using quadprog
    alpha = quadprog(H, f, [], [], Aeq, beq, lb, ub);
    
    % threshold support vectors due to numerical issues not reaching 0
    % SV_idx are all support vectors (non-zero alpha)
    % SV1_idx are alpha corresponding to vectors on the margin
    wx = sum(repmat(alpha.*trainlabels,1,m).*K,1)';
    SV_idx = find(alpha > 1e-6);
    SV1_idx = find(C > alpha & alpha > 1e-6);
    b = mean(trainlabels(SV1_idx) - wx(SV1_idx));
    SV = alpha(SV_idx);
    w = sum(repmat(alpha(SV_idx) .* trainlabels(SV_idx), 1, n) .* traindata(SV_idx,:), 1);
    
end

function K = rbf_kernel(traindata, gamma)
    % compute rbf_kernel
    [m, n] = size(traindata);
    
    for i = 1:m
        for q = 1:m
            K(i,q) = exp(-gamma*sum((traindata(i,:) - traindata(q,:)).^2));
        end
    end
end
