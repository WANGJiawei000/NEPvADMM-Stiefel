function [X_madmm, F_madmm, sparsity_madmm, time_madmm, error_madmm, iter_madmm, flag_succ] = MADMM_sPCA(B,option)

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

eta = 1e-2; rho = 1e2;
X = option.phi_init; Y = X;
Lambda = zeros(n,r);
F_ad = zeros(maxiter,1);
not_full_rank = 0;
flag_maxiter = 0;

for iter = 1:maxiter
    
    % X step: a Riemannian gradient step
    for i=1:100

        gx = -A*X + rho*(X - Y + Lambda);
        rgx = proj(X, gx);
        if norm(rgx, 'fro') < 1e-8
            break;
        end
        X = retr(X, -eta*rgx);
    end

    % Y step
    Y = wthresh(X + Lambda,'s', mu/rho);

    % Lambda step
    Lambda = Lambda + (X - Y);

    if type == 0 % data matrix
        AX = -(B'*(B*X));
    else
        AX = -(B*X);
    end

    F_ad(iter)= 0.5*sum(sum(X.*(AX)))+h(X); 

    if iter>2

        %normXY = norm(X-Y,'fro');
        %normX = norm(X,'fro');
        %normY = norm(Y,'fro');
        
        %if  normXY/max(1,max(normX,normY)) < tol
        if abs((F_ad(iter) - F_ad(iter-1))) <= tol

            break;

        end
    end
    
    if iter == maxiter
        flag_maxiter = 1;
    end
end

time_madmm= toc;

error_madmm = norm(Y - X, 'fro');

sparsity_madmm = sum(sum(abs(Y) <= 1e-6))/(n*r);

X_madmm = X;

if iter == maxiter
    flag_succ = 0; %fail
    F_madmm = 0;
    sparsity_madmm = 0;
    iter_madmm = 0;
    fprintf('MADMM fails to converge  \n');    
    time_madmm = 0;
else    
    flag_succ = 1; % success
    F_madmm = F_ad(iter);
    iter_madmm = iter;
    % residual_Q = norm(Q'*Q-eye(n),'fro')^2;
    fprintf('MADMM:Iter ***  Fval *** CPU  **** sparsity ********* err \n');
    
    print_format = ' %i     %1.5e    %1.2f     %1.2f            %1.3e \n';
    fprintf(1,print_format, iter, F_ad(iter), time_madmm, sparsity_madmm,  error_madmm);
end