function [X_nepv, F_nepv, sparsity_nepv, time_nepv, error_nepv, iter_nepv, flag_succ] = NEPvADMM_sPCA1(B,option)

tic;
r = option.r;
n = option.n;
mu = option.mu;
maxiter =option.maxiter;
tol = option.tol;
type = option.type;

if type==0 % data matrix
    A = B'*B;
else
    A = B;
end

h=@(X) mu*sum(sum(abs(X)));

rho = 250; gamma = 1e-8;
X = option.phi_init; Y = X; Z = X;
Lambda = zeros(n,r);
F_ad = zeros(maxiter,1);
not_full_rank = 0;
flag_maxiter = 0;

A_hat = A - rho*eye(n);
D = rho*Z - Lambda;

for iter = 1:maxiter

    % X step: NEPv step
    for i = 1:1

        DX = D*X';
        E = A_hat + DX + DX';
        [X_hat,~] = eigs(E,r,'largestreal');
        [U_hat,~,V_hat] = svd(X_hat'*D,"econ");
        X = X_hat*(U_hat*V_hat'); 

    end

    % Y step: Proximal Gradient
    XL = X + Lambda/rho;
    Y = wthresh(XL,'s',mu*(1+rho*gamma)/rho);

    Z = (Y/gamma + rho*XL) / (1/gamma + rho);

    % Lambda step:
    Lambda = Lambda + rho*(X - Z);

    D = rho*Z - Lambda;

    if type == 0 % data matrix
        AY = B'*(B*Y);
    else
        AY = B*Y;
    end

    F_ad(iter)= -0.5*sum(sum(Y.*(AY)))+h(Y); 

    if iter > 4

        if abs((F_ad(iter) - F_ad(iter-1))/F_ad(iter-1)) <= tol
            break
        end
    end
    
    if iter == maxiter
        flag_maxiter = 1;
    end
end

time_nepv = toc;

%error_nepv = norm(Z.'*Z - eye(r), 'fro');
error_nepv = norm(X - Y, 'fro');

%X((abs(X) <= 1e-5)) = 0;
X_nepv = X;

sparsity_nepv = sum(sum(abs(Z) <= 1e-5))/(n*r);

if iter > maxiter
    flag_succ = 0; %fail
    F_nepv = 0;
    sparsity_nepv = 0;
    iter_nepv = 0;
    fprintf('NEPvADMM fails to converge  \n');    
    time_nepv = 0;
else    
    flag_succ = 1; % success
    F_nepv = F_ad(iter);
    iter_nepv = iter;
    % residual_Q = norm(Q'*Q-eye(n),'fro')^2;
    %fprintf('NEPvADMM:Iter ***  Fval *** CPU  **** sparsity ********* err \n');
    
    %print_format = ' %i     %1.5e    %1.2f     %1.2f            %1.3e \n';
    %fprintf(1,print_format, iter, F_ad(iter), time_nepv, sparsity_nepv,  error_nepv);
end
