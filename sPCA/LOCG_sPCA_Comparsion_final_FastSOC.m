% Comparsion between NEPvADMM with ManPG and Subgradient and MADMM and PAMAL Method on SPCA
%% Let n = 100, 300; p = 5,10; \mu = 0.5,1
clc; close all; clear

addpath misc
addpath SSN_subproblem

%% Problem Generating
% Build function
f = @(X,H) -0.5*sum(sum(X.*(H*X)));
nabla_f = @(X,H) -H*X;
g = @(mu,X) mu*sum(sum(abs(X)));
F = @(mu,X,H) f(X,H) + g(mu,X);
sub_F = @(mu,X,H) - H*X + mu*sign(X);
partial_Xsub = @(X,H_hat,D) H_hat*X + D;

n_list = 2000; p_list = 20; mu_list = 1;

N = 1e3; iter1 = N; iter2 = N; iter3 = N; iter4 = N; iter5 = N; iter6 = N; avg = 1; 
rho = 15; gamma = 1e-8; 
eta = 5e-2; 
eta_m = 1e-2; rho_m = 100; 
eta_radmm = 1e-2; gamma_radmm = 1e-8; rho_radmm = 1e2; 

for n=n_list
    for p=p_list
        for mu=mu_list

            disp("test on n="+ n +" p="+ p+" mu="+mu);

            M = randn(n,n); M = (M+M')/2; [U1,~] = eig(M);
            v = rand(n,1) + 1e-6; H = U1*diag(v)*U1';

            H_hat = H - rho*eye(n);

            %% Algorithm
            % Parameters for ManPG subproblem
            L = abs(eigs(full(H),1)); % Lipschitz constant
            t = 1/L;
            inner_iter = 100;
            prox_fun = @(b,l,r) proximal_l1(b,l,r); % proximal function used in solving the subproblem
            t_min = 1e-4; % minimum stepsize
            num_linesearch = 0;
            num_inexact = 0; 
            inner_flag = 0;
            Dn = sparse(DuplicationM(p)); % vectorization for SSN
            pDn = (Dn'*Dn)\Dn'; % for SSN
            nu = 0.8; % penalty coefficient?
            alpha = 1; % stepsize fpr ManPG
            tol = 1e-8*n*p;

            % Parameters for SOC
            %rho_soc = svds(H,1)^2 + p/2;%  stepsize
            %rho_soc = svds(H,1)^2 + n*p*mu/10 + 1;
            rho_soc = svds(H,1)^2 + n/10; %n/50 ;% good for mu and r
            %rho_soc = 2* svds(H,1)^2  ;%  n/30 not converge   1.9* sometimes not converge
            lambda_soc = rho_soc;
            Ainv = inv(-H + (rho_soc + lambda_soc)*eye(n));

            
            F_val_manpg_avg = zeros([1,N]);
            F_val_subg_avg = zeros([1,N]);
            F_val_avg = zeros([1,N]);
            F_val_madmm_avg = zeros([1,N]);
            F_val_radmm_avg = zeros([1,N]);
            F_val_soc_avg = zeros([1,N]);

            cpu_time_manpg = zeros([avg,N]);
            cpu_time_admm = zeros([avg,N]);
            cpu_time_subg = zeros([avg, N]);
            cpu_time_madmm = zeros([avg, N]);
            cpu_time_radmm = zeros([avg, N]);
            cpu_time_soc = zeros([avg, N]);

            sparse_X = zeros([1, avg]);
            error_Y = zeros([1, avg]);
            sparse_U = zeros([1, avg]);
            sparse_W = zeros([1, avg]);
            error_U = zeros([1, avg]);
            error_W = zeros([1, avg]);
            sparse_Y_m = zeros([1, avg]);
            error_Y_m = zeros([1, avg]);
            sparse_Z_radmm = zeros([1, avg]);
            error_Z_radmm = zeros([1, avg]);
            sparse_X_soc = zeros([1, avg]);
            error_X_soc = zeros([1, avg]);


            for k = 1:avg
                if k == avg
                    disp("Total repitition " + k);
                end

                X = randn(n, p);
                X = orth(X);
                Y = X; Z = X; 
                Lambda = zeros(size(X));

                U = X; W = X; 
                X_m = X; Y_m = X; 
                X_radmm = X; Y_radmm = X; Z_radmm = X;
                P_soc = X; Q_soc = X; 
                Lambda_m = zeros(size(X_m));
                Lambda_radmm = zeros(size(X_radmm));
                Z_soc = zeros(n,p); b_soc = Z_soc; 

                D = rho*Z - Lambda;
                [U_start,~,V_start] = svd(X'*D,"econ");
                X = X*(U_start*V_start');

                HX = partial_Xsub(X,H_hat,D);
                XHX = X'*HX;
                RX = HX - X*(0.5*(XHX + XHX'));
                W_x = RX;
                W_x = W_x - X*(X'*W_x); W_x = orth(W_x); W_x = W_x - X*(X'*W_x); W_x = orth(W_x);
                W_x = [X, W_x]; m = size(W_x,2);
                Xz = eye(m); Xz = Xz(:,1:p);
                
                iter = 1;

                %% ManPG
                F_val_manpg_avg(1) = F_val_manpg_avg(1)+F(mu,U,H);

                for iter=2:N

                    manpg_start = tic;

                    neg_pg = H*U;

                    if alpha < t_min || num_inexact > 10
                        inner_tol = max(5e-16, min(1e-14,1e-5*tol*t^2)); % subproblem inexact;
                    else
                        inner_tol = max(1e-13, min(1e-11,1e-3*tol*t^2));
                    end

                    % The subproblem
                    %semi_newton = tic;
                    if iter == 2
                         [ PU,num_inner_x(iter),Lam_x, opt_sub_x(iter),in_flag] = Semi_newton_matrix(n,p,U,t,U + t*neg_pg,nu*t,inner_tol,prox_fun,inner_iter,zeros(p),Dn,pDn);
                        %      [ PY,num2(iter),r_norm(iter)]=fista(X,pgx,mu,t);
                    else
                         [ PU,num_inner_x(iter),Lam_x, opt_sub_x(iter),in_flag] = Semi_newton_matrix(n,p,U,t,U + t*neg_pg,nu*t,inner_tol,prox_fun,inner_iter,Lam_x,Dn,pDn);
                        %     [ PY,num2(iter),r_norm(iter)]=fista(X,pgx,mu,t);
                    end

                    %semi_newton_end = toc(semi_newton);
                    %cpu_time_newton(iter) = cpu_time_newton(iter)+semi_newton_end;

                    if in_flag == 1   % subprolem not exact.
                        inner_flag = 1 + inner_flag;
                    end

                    V = PU-U; % The V solved from SSN

                    % projection onto the Stiefel manifold
                    [T, SIGMA, S] = svd(PU'*PU);   SIGMA =diag(SIGMA);    U_temp = PU*(T*diag(sqrt(1./SIGMA))*S');

                    f_trial = f(U_temp,H);
                    F_trial = f_trial + g(mu,U_temp);   normV=norm(V,'fro');

                    %if  normD < tol 
                    %    break;
                    %end

                     %%% linesearch
                %     alpha_x = 1;
                %     while F_trial >= F_val(iter-1)-0.5/t*alpha_x*normV^2
                %         alpha_x = 0.5*alpha_x;
                %         linesearch_flag = 1;
                %         num_linesearch_x = num_linesearch_x + 1;
                %         if alpha_x < t_min
                %             num_inexact_x = num_inexact_x + 1;
                %             break;
                %         end
                %         PX = X+alpha_x*V;
                %         % projection onto the Stiefel manifold
                %         [U, SIGMA, S] = svd(PX'*PX);   SIGMA =diag(SIGMA);   X_temp = PX*(U*diag(sqrt(1./SIGMA))*S');
                %         f_trial = f(X_temp,H);
                %         F_trial = f_trial + lambda*h(X_temp);
                %     end
                %     X = X_temp; step_size_x(iter) = alpha_x;
                %     F_val(iter) = F_trial;
                %     norm_x(iter) = normV;

                    %%% Without linesearch
                    PU = U+alpha*V;
                    % projection onto the Stiefel manifold
                    [T, SIGMA, S] = svd(PU'*PU);   SIGMA =diag(SIGMA);   U_temp = PU*(T*diag(sqrt(1./SIGMA))*S');
                    U = U_temp; % update

                    elapsed_time_manpg = toc(manpg_start);

                    F_val_manpg(iter) = F(mu, U, H);
                    
                    if abs(F_val_manpg(iter) - F_val_manpg(iter-1)) <= 1e-6
                        break
                    end
                    
                    norm_x(iter) = normV;
                    norm_subg_ManPG(iter) = norm(proj(U, sub_F(mu, U, H)),'fro'); 

                    cpu_time_manpg(k,iter) = cpu_time_manpg(k,iter) + elapsed_time_manpg;

                    if iter < N
                        cpu_time_manpg(k,iter+1) = cpu_time_manpg(k,iter);
                    end

                    F_val_manpg_avg(iter) = F_val_manpg_avg(iter)+ F_val_manpg(iter);
                end
                iter1 = min(iter, iter1);


                %% NEPvADMM
                F_val_avg(1) = F_val_avg(1) + F(mu, X, H);

                for iter=2:N

                    admm_start = tic;

                    % X step: NEPv step 
                    X_old = X;

                    for i = 1:1

                        for j = 1:1

                            WZ = W_x*Xz;
                            DWZ = D*WZ';
                            E = W_x'*(H_hat + DWZ + DWZ')*W_x;

                            [Xz_hat,~] = eigs(E,p,'largestreal');

                            WZ_hat = W_x*Xz_hat;
                            [U_hat,~,V_hat] = svd(WZ_hat'*D,"econ");
                            Xz = Xz_hat*(U_hat*V_hat'); 

                        end

                        X = W_x*Xz;
                    end

                    % Y step: Proximal Gradient
                    XL = X + Lambda/rho;
                    Y = wthresh(XL,'s',mu*(1+rho*gamma)/rho);
                    Z = (Y/gamma + rho*XL) / (1/gamma + rho);

                    % Lambda step:
                    Lambda = Lambda + rho*(X - Z);

                    D = rho*Z - Lambda;

                    HX = partial_Xsub(X,H_hat,D);
                    XHX = X'*HX;
                    RX = HX - X*(0.5*(XHX + XHX'));
                    W_x = [RX X_old];
                    %W_x = RX;
                    W_x = W_x - X*(X'*W_x); W_x = orth(W_x); W_x = W_x - X*(X'*W_x); W_x = orth(W_x);
                    W_x = [X, W_x]; m = size(W_x,2);
                    Xz = eye(m); Xz = Xz(:,1:p);
                                      
                    elapsed_time_admm = toc(admm_start);

                    % Value update
                    F_val(iter) = F(mu, X, H);

                    if abs((F_val(iter) - F_val(iter-1))/F_val(iter-1)) <= 1e-5
                        break
                    end

                    cpu_time_admm(k,iter) = cpu_time_admm(k,iter) + elapsed_time_admm;

                    if iter < N
                        cpu_time_admm(k,iter+1) = cpu_time_admm(k,iter);
                    end

                    F_val_avg(iter) = F_val_avg(iter) + F_val(iter);
                end
                iter2 = min(iter, iter2);


                %% MADMM
                F_val_madmm_avg(1) = F_val_madmm_avg(1) + F(mu, X_m, H);
                
                for iter=2:N

                    madmm_start = tic;

                    % X step: a Riemannian gradient step
                    for i=1:100
                        gx_m = -H*X_m + rho_m*(X_m - Y_m + Lambda_m);
                        rgx_m = proj(X_m, gx_m);
                        if norm(rgx_m, 'fro') < 1e-8
                            break;
                        end
                        X_m = retr(X_m, -eta_m*rgx_m);
                    end

                    % Y step
                    Y_m = wthresh(X_m + Lambda_m,'s', mu/rho_m);

                    % Lambda step
                    Lambda_m = Lambda_m + (X_m - Y_m);

                    elapsed_time_madmm = toc(madmm_start);

                    % Value update
                    F_val_madmm(iter) = F(mu, X_m, H);
                    F_val_madmm_avg(iter) = F_val_madmm_avg(iter) + F_val_madmm(iter);

                    if abs(F_val_madmm(iter) - F_val_madmm(iter-1)) <= 1e-6
                        break
                    end

                    cpu_time_madmm(k,iter) = cpu_time_madmm(k,iter) + elapsed_time_madmm;

                    if iter < N
                        cpu_time_madmm(k,iter+1) = cpu_time_madmm(k,iter);
                    end

                end
                iter4 = min(iter, iter4);


                %% RADMM
                F_val_radmm_avg(1) = F_val_radmm_avg(1) + F(mu, X_radmm, H);

                for iter=2:N

                    radmm_start = tic;

                    % X step: a gradient step
                    for i=1:1
                        gx_radmm = -H*X_radmm + Lambda_radmm + rho_radmm*(X_radmm - Z_radmm);
                        rgx_radmm = proj(X_radmm, gx_radmm);
                        X_radmm = retr(X_radmm, -(eta_radmm)*rgx_radmm);
                    end

                    % Z step (also update Y)
                    Y_radmm = wthresh(X_radmm + Lambda_radmm/rho_radmm,'s',mu*(1+rho_radmm*gamma_radmm)/rho_radmm);
                    Z_radmm = (Y_radmm/gamma_radmm + Lambda_radmm + rho_radmm*X_radmm) / (1/gamma_radmm + rho_radmm);

                    % Lambda step
                    Lambda_radmm = Lambda_radmm + rho_radmm*(X_radmm - Z_radmm);

                    elapsed_time_radmm = toc(radmm_start);

                    % Value update
                    F_val_radmm(iter) = F(mu, X_radmm, H);

                    if abs(F_val_radmm(iter) - F_val_radmm(iter-1)) <= 1e-6
                        break
                    end

                    cpu_time_radmm(k,iter) = cpu_time_radmm(k,iter) + elapsed_time_radmm;

                    if iter < N
                        cpu_time_radmm(k,iter+1) = cpu_time_radmm(k,iter);
                    end

                    F_val_radmm_avg(iter) = F_val_radmm_avg(iter) + F_val_radmm(iter);
                end
                iter5 = min(iter, iter5);


                %% SOC
                F_val_soc(1) = F(mu, P_soc, H);
                F_val_soc_avg(1) = F_val_soc_avg(1) + F_val_soc(1);
                
                for iter=2:N

                    admm_soc_start = tic;

                    LZ_soc = rho_soc*(P_soc - Z_soc) + lambda_soc*(Q_soc - b_soc);
                
                    % X is P in paper
                    X_soc = Ainv*LZ_soc;
                
                    % Q is Q in paper
                    Q_soc = sign(X_soc + b_soc).*max(0, abs(X_soc + b_soc)-mu/lambda_soc);
                    
                    %%%% solve P
                    
                    Y_soc = X_soc + Z_soc;
                
                    [U_soc,~,V_soc] = svd(Y_soc,0);
                
                    % P is X in paper
                    P_soc = U_soc*V_soc';
                
                    % Larange Multipliers
                    Z_soc  = Z_soc + X_soc - P_soc;
                    b_soc  = b_soc + X_soc - Q_soc;
                                   
                    elapsed_time_soc = toc(admm_soc_start);

                    % Value update
                    F_val_soc(iter) = F(mu, P_soc, H);
                    F_val_soc_avg(iter) = F_val_soc_avg(iter) + F_val_soc(iter);

                    if iter > 2
            
                        normXQ = norm(X_soc - Q_soc,'fro');
                        normQ = norm(Q_soc,'fro');
                        normX = norm(X_soc,'fro');
                        normP = p;
                        normXP = norm(X_soc - P_soc,'fro');
                
                        if  normXQ/max(1,max(normQ,normX)) + normXP/max(1,max(normP,normX)) < 1e-6
                            if  abs(F_val_soc(iter) - F_val_soc(iter-1)) <= 1e-8
                                break;
                            end
                        end
                    end

                    cpu_time_soc(k,iter) = cpu_time_soc(k,iter) + elapsed_time_soc;
                    
                    if iter < N
                        cpu_time_soc(k,iter+1) = cpu_time_soc(k,iter);
                    end
                end
                iter6 = min(iter, iter6);


                sparse_X(k) = sum(sum(abs(Z) <= 1e-8))/(n*p);
                error_Y(k) = norm(Z.'*Z - eye(p), 'fro');

                sparse_U(k) = sum(sum(abs(U) <= 1e-8))/(n*p);
                sparse_W(k) = sum(sum(abs(W) <= 1e-8))/(n*p);
                error_U(k) = norm(U.'*U - eye(p), 'fro');
                error_W(k) = norm(W.'*W - eye(p), 'fro');
                sparse_Y_m(k) = sum(sum(abs(Y_m) <= 1e-8))/(n*p);
                error_Y_m(k) = norm(Y_m.'*Y_m - eye(p), 'fro');

                sparse_Z_radmm(k) = sum(sum(abs(Z_radmm) <= 1e-8))/(n*p);
                error_Z_radmm(k) = norm(Z_radmm.'*Z_radmm - eye(p), 'fro');

                sparse_X_soc(k) = sum(sum(abs(Y_soc) <= 1e-8))/(n*p);
                error_X_soc(k) = norm(Y_soc.'*Y_soc - eye(p), 'fro');
            end 

            F_val_avg = (F_val_avg/avg);
            F_val_manpg_avg = (F_val_manpg_avg/avg);
            F_val_madmm_avg = (F_val_madmm_avg/avg);
            F_val_radmm_avg = (F_val_radmm_avg/avg);
            F_val_soc_avg = (F_val_soc_avg/avg);

            cpu_time_admm = sum(cpu_time_admm,1)/avg;
            cpu_time_manpg = sum(cpu_time_manpg,1)/avg;
            cpu_time_madmm = sum(cpu_time_madmm,1)/avg;
            cpu_time_radmm = sum(cpu_time_radmm,1)/avg;
            cpu_time_soc = sum(cpu_time_soc,1)/avg;

            av_sparse_x = sum(sparse_X)/avg;
            av_sparse_u = sum(sparse_U)/avg;
            av_sparse_y_m = sum(sparse_Y_m)/avg;
            av_sparse_z_radmm = sum(sparse_Z_radmm)/avg;
            av_sparse_y_soc = sum(sparse_X_soc)/avg;
            
            av_error_y = sum(error_Y)/avg;
            av_error_u = sum(error_U)/avg;
            av_error_y_m = sum(error_Y_m)/avg;
            av_error_z_radmm = sum(error_Z_radmm)/avg;
            av_error_x_soc = sum(error_X_soc)/avg;

            disp("sparisty for manpg, SOC, MADMM, RADMM and NEPvADMM: ")

            disp([av_sparse_u, av_sparse_y_soc, av_sparse_y_m, av_sparse_z_radmm, av_sparse_x])
            
            disp("error of manpg, SOC, MADMM, RADMM and NEPvADMM: ")
            
            disp([av_error_u, av_error_x_soc, av_error_y_m, av_error_z_radmm, av_error_y])
            
            disp("CPU time for manpg, SOC, MADMM, RADMM and NEPvADMM: ")
            
            disp([cpu_time_manpg(iter1 - 1), cpu_time_soc(iter6 - 1), cpu_time_madmm(iter4 - 1), cpu_time_radmm(iter5 - 1), cpu_time_admm(iter2 - 1)]);
            
            disp("function value for output manpg, SOC, MADMM, RADMM and NEPvADMM: ")

            disp([F_val_manpg_avg(iter1 - 1), F_val_soc_avg(iter6 - 1), F_val_madmm_avg(iter4 - 1), F_val_radmm_avg(iter5 - 1), F_val_avg(iter2 - 1)]);

%%
            figure4 = figure(4);
            clf
            semilogy(cpu_time_manpg(1:iter1), F_val_manpg_avg(1:iter1),LineWidth = 2); hold on;
            semilogy(cpu_time_admm(1:iter2), F_val_avg(1:iter2),LineWidth = 2); hold on;
            semilogy(cpu_time_madmm(1:iter4), F_val_madmm_avg(1:iter4),LineWidth = 2); hold on;
            semilogy(cpu_time_radmm(1:iter5), F_val_radmm_avg(1:iter5),LineWidth = 2); hold on;
            semilogy(cpu_time_soc(1:iter6), F_val_soc_avg(1:iter6),LineWidth = 2); hold on;
            xlabel("CPU time", 'FontSize', 18);
            ylabel("Objective value", 'FontSize', 18);
            labels2 = {'ManPG','accNEPvADMM','MADMM', 'RADMM', 'SOC'};
            legend(labels2, 'Interpreter', 'latex', 'FontSize', 11);
            %filename = "grid_search_plots/n_" + n + "_p_" + p + "_mu_" + mu + "_time_fval.pdf";
            %saveas(figure4, filename); 

            figure5 = figure(5);
            clf
            plot((1:1:iter1-1), F_val_manpg_avg(1:iter1-1),LineWidth = 2); hold on;
            plot((1:1:iter2-1), F_val_avg(1:iter2-1),LineWidth = 2); hold on;
            plot((1:1:iter4-1), F_val_madmm_avg(1:iter4-1),LineWidth = 2); hold on;
            plot((1:1:iter5-1), F_val_radmm_avg(1:iter5-1),LineWidth = 2); hold on;
            plot((1:1:iter6-1), F_val_soc_avg(1:iter6-1),LineWidth = 2); hold on;
            xlabel("Iteration Count", 'FontSize', 18)
            ylabel("Objective value", 'FontSize', 18);
            labels1 = {'ManPG','accNEPvADMM','MADMM', 'RADMM', 'SOC'};
            legend(labels1, 'Interpreter', 'latex', 'FontSize', 11);
            %filename = "grid_search_plots/n_" + n + "_p_" + p + "_mu_" + mu + "_time_fval.pdf";
            %saveas(figure4, filename); 
%%            
        end

    end
end
