function yhat = predictSvmRbfKernel(SV, alpha, trainlabels, traindata, testdata, b, gamma)
    
    [m,n] = size(testdata);
    K = rbf_kernel(traindata, testdata, gamma);
    wx = sum(repmat(alpha .* trainlabels, 1, m) .* K , 1);
    yhat = sign(wx + b);
end

function K = rbf_kernel(traindata, testdata, gamma)
    % compute rbf_kernel
    [m_t, n_t] = size(traindata);
    [m, n] = size(testdata);
    
    for i = 1:m_t
        for q = 1:m
            K(i,q) = exp(-gamma*sum((traindata(i,:) - testdata(q,:)).^2));
        end
    end
end