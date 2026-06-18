function [X_Soc, F_SOC, sparsity_soc, time_soc, error_XPQ, iter_soc, flag_succ] = SOC_sPCA(B,option)

tic;
r = option.r;
n = option.n;
mu = option.mu;
maxiter =option.maxiter;
tol = option.tol;
type = option.type;

if type==0 % data matrix
    A = -B'*B;
else
    A = -B;
end

h=@(X) mu*sum(sum(abs(X)));

%rho = svds(B,1)^2 + r/2;%  stepsize
%rho = svds(B,1)^2 + n*r*mu/25 + 1;
rho = svds(B,1)^2 + n/7; %n/50 ;% good for mu and r
%rho = 2* svds(B,1)^2  ;%  n/30 not converge   1.9* sometimes not converge
lambda = rho;
P = option.phi_init;    Q = P;
Z = zeros(n,r); 
b=Z;  
F_ad=zeros(maxiter,1);
not_full_rank = 0;

%chA = chol( 2*A + (r+lambda)*eye(d));
Ainv = inv(A + (rho+lambda)*eye(n));
flag_maxiter = 0;

for itera=1:maxiter

    LZ = rho*(P-Z)+lambda*(Q-b);
    %   X=A_bar\LB;
    %  X = chA\(chA'\LZ);
    % X is P in paper
    X = Ainv*LZ;
    %%%% shrinkage Q
    % Q is Q in paper
    Q = sign(X+b).*max(0,abs(X+b)-mu/lambda);
    
    %%%% solve P
    
    Y = X + Z;
    %%%%%%%%%%%%%   svd Y'*Y
    %     [U, D, S] = svd(Y'*Y);
    %     D = diag(D);
    %     if abs(prod(D))>0
    %         P = Y*(U*diag(sqrt(1./D))*S');
    %     else
    %         not_full_rank = not_full_rank+1;
    %     end
    [U,~,V]= svd(Y,0);
    % P is X in paper
    P = U*V';
    %%%%%%%%%
    % Larange Multipliers
    Z  = Z+X-P;
    b  = b+X-Q;

    if type == 0 % data matrix
        AP = B'*(B*P);
    else
        AP = B*P;
    end
    F_ad(itera)= -0.5*sum(sum(P.*(AP)))+h(P);    
    
    if itera>2
        normXQ = norm(X-Q,'fro');
        normQ = norm(Q,'fro');
        normX = norm(X,'fro');
        normP = r;
        normXP = norm(X-P,'fro');
        if  normXQ/max(1,max(normQ,normX)) + normXP/max(1,max(normP,normX)) < tol
            if  abs((F_ad(itera) - F_ad(itera-1))) <= 1e-8
                break;
            end
        end
    end
    
    P_old=P;
    if itera ==maxiter
        flag_maxiter =1;
    end
end

%P((abs(P)<=1e-5))=0;
X_Soc=P;

time_soc= toc;

error_XPQ = norm(X-P,'fro') + norm(X-Q,'fro');
sparsity_soc = sum(sum(abs(P) <= 1e-6))/(n*r);

if itera == maxiter
    flag_succ = 0; %fail
    F_SOC = 0;
    sparsity_soc = 0;
    iter_soc = 0;
    fprintf('SOC fails to converge  \n');
    
    %fprintf('Soc:Iter ***  Fval *** CPU  **** sparsity ********* err \n');
    
    %print_format = ' %i     %1.5e    %1.2f     %1.2f            %1.3e \n';
    %fprintf(1,print_format, itera, F_ad(itera), time_soc, sparsity_soc,  error_XPQ);
    time_soc = 0;
else    
    flag_succ = 1; % success
    F_SOC = F_ad(itera);
    iter_soc = itera;
    % residual_Q = norm(Q'*Q-eye(n),'fro')^2;
    fprintf('Soc:Iter ***  Fval *** CPU  **** sparsity ********* err \n');
    
    print_format = ' %i     %1.5e    %1.2f     %1.2f            %1.3e \n';
    fprintf(1,print_format, itera, F_ad(itera), time_soc, sparsity_soc,  error_XPQ);
end