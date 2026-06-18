% Compare how the sparsity, fv, error changes with regard to mu for sPCA
clc; close all; clear

addpath misc
addpath SSN_subproblem

n_set=[100; 150; 200; 250; 300; 500; 1000];
r_set = [2; 5; 10; 15; 30];
mu_set = [0.1:0.5:5, 5];


test_number = 30;


index = 1;

for id_mu = 1:size(mu_set,2)

    mu = mu_set(id_mu);
    fid =1;
    
    r1 = r_set(2); n1 = n_set(6);
    r2 = r_set(4); n2 = n_set(6);
    r3 = r_set(2); n3 = n_set(3);
    r4 = r_set(4); n4 = n_set(3);
    
    succ_no_nepv1 = 0; fail_no_nepv1 = 0;
    succ_no_nepv2 = 0; fail_no_nepv2 = 0;
    succ_no_nepv3 = 0; fail_no_nepv3 = 0;
    succ_no_nepv4 = 0; fail_no_nepv4 = 0;

    for test_random = 1:test_number %times average.

        %fprintf(fid,'==============================================================================================\n');
        
        %rng('shuffle');

        M1 = randn(n1,n1); M1 = (M1+M1')/2; [U11,~] = eig(M1);
        v1 = rand(n1,1) + 1e-6; B1 = U11*diag(v1)*U11';
        type1 = 1;

        M2 = randn(n2,n2); M2 = (M2+M2')/2; [U12,~] = eig(M2);
        v2 = rand(n2,1) + 1e-6; B2 = U12*diag(v2)*U12';
        type2 = 1;

        M3 = randn(n3,n3); M3 = (M3+M3')/2; [U13,~] = eig(M3);
        v3 = rand(n3,1) + 1e-6; B3 = U13*diag(v3)*U13';
        type3 = 1;

        M4 = randn(n4,n4); M4 = (M4+M4')/2; [U14,~] = eig(M4);
        v4 = rand(n4,1) + 1e-6; B4 = U14*diag(v4)*U14';
        type4 = 1;

        %fprintf(fid,'- n -- r -- mu --------\n');
        %fprintf(fid,'%4d %3d %3.3f \n',n,r,mu);
        %fprintf(fid,'----------------------------------------------------------------------------------\n');
        
        %rng('shuffle');


        phi_init1 = randn(n1, r1);
        phi_init1 = orth(phi_init1);

        phi_init2 = randn(n2, r2);
        phi_init2 = orth(phi_init2);

        phi_init3 = randn(n3, r3);
        phi_init3 = orth(phi_init3);

        phi_init4 = randn(n4, r4);
        phi_init4 = orth(phi_init4);

        
        %%%%%% NEPvADMM parameter
        option_nepv1.phi_init = phi_init1; option_nepv1.maxiter = 1e3; option_nepv1.tol = 1e-5;
        option_nepv1.r = r1;   option_nepv1.n = n1;  option_nepv1.mu = mu;
        option_nepv1.type = type1;

        [X_nepv1, F_nepv1(test_random), sparsity_nepv1(test_random), time_nepv1(test_random),...
            error_nepv1(test_random), maxit_att_nepv1(test_random), succ_flag_nepv1] = NEPvADMM_sPCA1(B1,option_nepv1);

        if succ_flag_nepv1 == 1
            succ_no_nepv1 = succ_no_nepv1 + 1;
        end

        if succ_flag_nepv1 == 0
            fail_no_nepv1 = fail_no_nepv1 + 1;
        end

        %%%%%% NEPvADMM parameter
        option_nepv2.phi_init = phi_init2; option_nepv2.maxiter = 1e3; option_nepv2.tol = 1e-5;
        option_nepv2.r = r2;   option_nepv2.n = n2;  option_nepv2.mu = mu;
        option_nepv2.type = type2;

        [X_nepv2, F_nepv2(test_random), sparsity_nepv2(test_random), time_nepv2(test_random),...
            error_nepv2(test_random), maxit_att_nepv2(test_random), succ_flag_nepv2] = NEPvADMM_sPCA1(B2,option_nepv2);

        if succ_flag_nepv2 == 1
            succ_no_nepv2 = succ_no_nepv2 + 1;
        end

        if succ_flag_nepv2 == 0
            fail_no_nepv2 = fail_no_nepv2 + 1;
        end


        %%%%%% NEPvADMM parameter
        option_nepv3.phi_init = phi_init3; option_nepv3.maxiter = 1e3; option_nepv3.tol = 1e-5;
        option_nepv3.r = r3;   option_nepv3.n = n3;  option_nepv3.mu = mu;
        option_nepv3.type = type3;

        [X_nepv3, F_nepv3(test_random), sparsity_nepv3(test_random), time_nepv3(test_random),...
            error_nepv3(test_random), maxit_att_nepv3(test_random), succ_flag_nepv3] = NEPvADMM_sPCA1(B3,option_nepv3);

        if succ_flag_nepv3 == 1
            succ_no_nepv3 = succ_no_nepv3 + 1;
        end

        if succ_flag_nepv3 == 0
            fail_no_nepv3 = fail_no_nepv3 + 1;
        end
       

        %%%%%% NEPvADMM parameter
        option_nepv4.phi_init = phi_init4; option_nepv4.maxiter = 1e3; option_nepv4.tol = 1e-5;
        option_nepv4.r = r4;   option_nepv4.n = n4;  option_nepv4.mu = mu;
        option_nepv4.type = type4;

        [X_nepv4, F_nepv4(test_random), sparsity_nepv4(test_random), time_nepv4(test_random),...
            error_nepv4(test_random), maxit_att_nepv4(test_random), succ_flag_nepv4] = NEPvADMM_sPCA1(B4,option_nepv4);

        if succ_flag_nepv4 == 1
            succ_no_nepv4 = succ_no_nepv4 + 1;
        end

        if succ_flag_nepv4 == 0
            fail_no_nepv4 = fail_no_nepv4 + 1;
        end

    end
                    
    index = index + 1;
    
    iter1.nepv1(id_mu) =  sum(maxit_att_nepv1)/succ_no_nepv1;
    time1.nepv1(id_mu) =  sum(time_nepv1)/succ_no_nepv1;
    Fval1.nepv1(id_mu) =  sum(F_nepv1)/succ_no_nepv1;
    Sp1.nepv1(id_mu) =  sum(sparsity_nepv1)/succ_no_nepv1;
    Err1.nepv1(id_mu) = sum(error_nepv1)/succ_no_nepv1;
    
    iter2.nepv2(id_mu) =  sum(maxit_att_nepv2)/succ_no_nepv2;
    time2.nepv2(id_mu) =  sum(time_nepv2)/succ_no_nepv2;
    Fval2.nepv2(id_mu) =  sum(F_nepv2)/succ_no_nepv2;
    Sp2.nepv2(id_mu) =  sum(sparsity_nepv2)/succ_no_nepv2;
    Err2.nepv2(id_mu) = sum(error_nepv2)/succ_no_nepv2;

    iter3.nepv3(id_mu) =  sum(maxit_att_nepv3)/succ_no_nepv3;
    time3.nepv3(id_mu) =  sum(time_nepv3)/succ_no_nepv3;
    Fval3.nepv3(id_mu) =  sum(F_nepv3)/succ_no_nepv3;
    Sp3.nepv3(id_mu) =  sum(sparsity_nepv3)/succ_no_nepv3;
    Err3.nepv3(id_mu) = sum(error_nepv3)/succ_no_nepv3;

    iter4.nepv4(id_mu) =  sum(maxit_att_nepv4)/succ_no_nepv4;
    time4.nepv4(id_mu) =  sum(time_nepv4)/succ_no_nepv4;
    Fval4.nepv4(id_mu) =  sum(F_nepv4)/succ_no_nepv4;
    Sp4.nepv4(id_mu) =  sum(sparsity_nepv4)/succ_no_nepv4;
    Err4.nepv4(id_mu) = sum(error_nepv4)/succ_no_nepv4;

end


%% plot
figure(1);
plot(mu_set, time1.nepv1,'m-o','MarkerSize',10,'linewidth',2); hold on;
plot(mu_set, time2.nepv2,'g-+','MarkerSize',10,'linewidth',2); hold on;
plot(mu_set, time3.nepv3,'c-*','MarkerSize',10,'linewidth',2); hold on;
plot(mu_set, time4.nepv4,'r-x','MarkerSize',10,'linewidth',2);
legend('n = 500,p = 5','n = 500,p = 15','n = 200, p = 5', 'n = 200, p = 15', 'FontSize', 11);
xlabel('\mu', 'FontSize', 18);   ylabel('CPU', 'FontSize', 18);


figure(2)
semilogy(mu_set, Fval1.nepv1, 'm-o','MarkerSize',10,'linewidth',2); hold on;
semilogy(mu_set, Fval2.nepv2, 'g-+','MarkerSize',10,'linewidth',2); hold on;
semilogy(mu_set, Fval3.nepv3, 'c-*','MarkerSize',10,'linewidth',2); hold on;
semilogy(mu_set, Fval4.nepv4, 'r-x','MarkerSize',10,'linewidth',2);
legend('n = 500,p = 5','n = 500,p = 15','n = 200, p = 5','n = 200, p = 15', 'Location','southeast', 'FontSize', 11);
xlabel('\mu', 'FontSize', 18);   ylabel('objective value', 'FontSize', 18);


figure(3)
plot(mu_set, Sp1.nepv1, 'm-o','MarkerSize',10,'linewidth',2); hold on;
plot(mu_set, Sp2.nepv2, 'g-+','MarkerSize',10,'linewidth',2); hold on;
plot(mu_set, Sp3.nepv3, 'c-*','MarkerSize',10,'linewidth',2); hold on;
plot(mu_set, Sp4.nepv4, 'r-x','MarkerSize',10,'linewidth',2);
legend('n = 500,p = 5','n = 500,p = 15','n = 200, p = 5','n = 200, p = 15','Location','southeast', 'FontSize', 11);
xlabel('\mu', 'FontSize', 18);   ylabel('sparsity', 'FontSize', 18);


figure(4)
plot(mu_set, iter1.nepv1, 'm-o','MarkerSize',10,'linewidth',2); hold on;
plot(mu_set, iter2.nepv2, 'g-+','MarkerSize',10,'linewidth',2); hold on;
plot(mu_set, iter3.nepv3, 'c-*','MarkerSize',10,'linewidth',2); hold on;
plot(mu_set, iter4.nepv4, 'r-x','MarkerSize',10,'linewidth',2);
legend('n = 500,p = 5','n = 500,p = 15','n = 200, p = 5','n = 200, p = 15', 'FontSize', 11);
xlabel('\mu', 'FontSize', 18);   ylabel('number of iterations', 'FontSize', 18);


figure(5)
plot(mu_set, Err1.nepv1, 'm-o','MarkerSize',10,'linewidth',2); hold on;
plot(mu_set, Err2.nepv2, 'g-+','MarkerSize',10,'linewidth',2); hold on;
plot(mu_set, Err3.nepv3, 'c-*','MarkerSize',10,'linewidth',2); hold on;
plot(mu_set, Err4.nepv4, 'r-x','MarkerSize',10,'linewidth',2);
legend('n = 500,p = 5','n = 500,p = 15','n = 200, p = 5','n = 200, p = 15', 'FontSize', 11);
xlabel('\mu', 'FontSize', 18);   ylabel('||X_{k} - Z_{k}||_{F}', 'FontSize', 18);
