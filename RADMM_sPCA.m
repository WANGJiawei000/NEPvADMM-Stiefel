function [X_radmm, F_radmm, sparsity_radmm, time_radmm, error_radmm, iter_radmm, flag_succ]=RADMM_sPCA(B,option)

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

eta = 1e-2; gamma = 1e-8; rho = 1e2;
X = option.phi_init; Y = X; Z = X;
Lambda = zeros(n,r);
F_ad = zeros(maxiter,1);
not_full_rank = 0;
flag_maxiter = 0;

for iter = 1:maxiter

    % X step: a gradient step
    for i=1:1
        gx = -A*X + Lambda + rho*(X - Z);
        rgx = proj(X, gx);
        X = retr(X, -(eta)*rgx);
    end

    % Z step (also update Y)
    Y = wthresh(X + Lambda/rho,'s',mu*(1+rho*gamma)/rho);
    Z = (Y/gamma + Lambda + rho*X) / (1/gamma + rho);

    % Lambda step
    Lambda = Lambda + rho*(X - Z);

    if type == 0 % data matrix
        AX = B'*(B*X);
    else
        AX = B*X;
    end

    F_ad(iter)= -0.5*sum(sum(X.*(AX)))+h(X); 

    if iter>2

        %{
        normXZ = norm(X-Z,'fro');
        normX = norm(X,'fro');
        normZ = norm(Z,'fro');
        
        if  normXZ/max(1,max(normX,normZ)) < tol

            break;

        end
        %}
        if abs(F_ad(iter) - F_ad(iter-1)) <= tol
            break;
        end
    end
    
    if iter == maxiter
        flag_maxiter = 1;
    end
end

time_radmm= toc;

error_radmm = norm(X.'*X - eye(r), 'fro');
sparsity_radmm = sum(sum(abs(X) <= 1e-6))/(n*r);

%X((abs(X) <= 1e-5)) = 0;
X_radmm = X;

%sparsity_radmm = sum(sum(X == 0))/(n*r);

if iter == maxiter
    flag_succ = 0; %fail
    F_radmm = 0;
    sparsity_radmm = 0;
    iter_radmm = 0;
    fprintf('RADMM fails to converge  \n');    
    time_radmm = 0;
else    
    flag_succ = 1; % success
    F_radmm = F_ad(iter);
    iter_radmm = iter;
    % residual_Q = norm(Q'*Q-eye(n),'fro')^2;
    fprintf('RADMM:Iter ***  Fval *** CPU  **** sparsity ********* err \n');
    
    print_format = ' %i     %1.5e    %1.2f     %1.2f            %1.3e \n';
    fprintf(1,print_format, iter, F_ad(iter), time_radmm, sparsity_radmm,  error_radmm);
end