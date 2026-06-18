n = 2000 ; p = 5;

%% sPCA data matrix generation
M = randn(n,n); M = (M+M')/2; [U1,~] = eig(M);
v = rand(n,1) + 1e-6; H = U1*diag(v)*U1';

%% sPCA initial variable generation
X = randn(n, p);
X = orth(X);