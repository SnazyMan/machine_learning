function yhat = predictSvmQuadKernel(SV, alpha, trainlabels, traindata, testdata, b, q)
    
    [m,n] = size(testdata);
    K = (traindata * testdata' + 1).^q;
    wx = sum(repmat(alpha .* trainlabels, 1, m) .* K , 1);
    yhat = sign(wx + b);
end