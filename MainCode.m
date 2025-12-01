
A=[1.01,0.01,0;0.01,1.01,0.01;0,0.01,1.01];% system matrix A
B=1*eye(3); % input matrix B
Q=0.001*eye(3); % weighting matrix Q
R=1*eye(3); % weighting matrix R
[Pstar,Kstar,L,info]=idare(A,B,Q,R);
Kstar=-Kstar;
n=size(A,1);
m=size(B,2);
Sigma_x = eye(n);
Sigma_u = eye(m);
Sigma_w = 0.1*eye(n);
% Data collection
N = 1;        
K = 100;         
number=30;
iteration=36;

[X_t,U_t,X_t1]=datacollection(K,N,m,n,Sigma_x,Sigma_u,Sigma_w,A,B);

X_t_reshaped  = reshape(X_t,  n,  N*K);
U_t_reshaped  = reshape(U_t,  m,  N*K);
X_t1_reshaped = reshape(X_t1, n,  N*K);
W=compute_W(Sigma_w);



% Algorithm params
params.E = K;
e_vec = 1:params.E;
params.eta = 0.001*sqrt(e_vec);        
params.lambda =0.001*sqrt(e_vec); 
params.zeta = (e_vec - 1) ./ e_vec;  
% ball radii
r_y = 1;      % radius for y ball constraint
r_xi = 1;     % radius for xi ball constraint

% Algorithm parameter
x0 = randn(n,1); 
u0 = randn(m,1); 
x10 = A*x0 + B*u0 + 0.01*randn(n,1);
dummyGamma = compute_Gamma_hat(x0, u0, K, x10, W);
xi0 = zeros(length(dummyGamma),1);
y0  = zeros(N,1);
opts.verbose = true;


khistoryGN=zeros(m,n,iteration);
phistoryGN=zeros(m,m,iteration);
chistoryGN=zeros(iteration,number);

khistoryNP=zeros(m,n,iteration);
phistoryNP=zeros(m,m,iteration);
chistoryNP=zeros(iteration,number);


khistoryGNL=zeros(m,n,iteration);
phistoryGNL=zeros(m,m,iteration);
chistoryGNL=zeros(iteration,number);

khistoryNPL=zeros(m,n,iteration);
phistoryNPL=zeros(m,m,iteration);
chistoryNPL=zeros(iteration,number);

khistoryGNL2=zeros(m,n,iteration);
phistoryGNL2=zeros(m,m,iteration);
chistoryGNL2=zeros(iteration,number);

khistoryGNL3=zeros(m,n,iteration);
phistoryGNL3=zeros(m,m,iteration);
chistoryGNL3=zeros(iteration,number);


khistoryNPL2=zeros(m,n,iteration);
phistoryNPL2=zeros(m,m,iteration);
chistoryNPL2=zeros(iteration,number);

[~,Kinitial]=idare(A,B,100*Q,R);
factorGNM=0.05;
esterror=(zeros(iteration,1));
for j=1:number
    KcurrentGN=-Kinitial;
KcurrentNP=-Kinitial;
KcurrentGNL=-Kinitial;
KcurrentNPL=-Kinitial;
KcurrentGNL2=-Kinitial;
KcurrentNPL2=-Kinitial;
KcurrentGNL3=-Kinitial;
 %disp(["numbr" ,num2str(j),"iteratio",num2str(i)])
for i=1:iteration
    disp(["numbr" ,num2str(j),"iteratio",num2str(i)])
    %% GN CPD
   % Record the data
   khistoryGN(:,:,i)=KcurrentGN;
   phistoryGN(:,:,i)=dlyap((A+B*KcurrentGN)',Q+KcurrentGN'*R*KcurrentGN); 
   chistoryGN(i,j)=trace(phistoryGN(:,:,i));
   % NPG or GNM
   [xi_bar, y_bar, history] = conditional_stochastic_primal_dual_ball(KcurrentGN, X_t, U_t, X_t1, xi0, y0, params, r_xi, r_y, W, Q, R, opts);
   [BP_A_rec, BP_B_rec, P_K_rec] = xiK_to_matrices(xi_bar, n, m);
   eta=1;
   KcurrentGN=KcurrentGN-factorGNM*eta*(KcurrentGN+inv(R+BP_B_rec)*(BP_A_rec'));
   bpa=(B'*phistoryGN(:,:,i)*A)';
   xi_ref=[bpa(:);vecs(B'*phistoryGN(:,:,i)*B);vecs(phistoryGN(:,:,i))];
   esterror(i,:)=norm(xi_ref);
 %% GN LS
   % Record the data
   khistoryGNL(:,:,i)=KcurrentGNL;
   phistoryGNL(:,:,i)=dlyap((A+B*KcurrentGNL)',Q+KcurrentGNL'*R*KcurrentGNL); 
   chistoryGNL(i,j)=trace(phistoryGNL(:,:,i));
     [Hk, bk] = generate_Hb(X_t_reshaped, U_t_reshaped, X_t1_reshaped, KcurrentGNL, Q, R, W);
     xi_bar=Hk \ bk;
   
   [BP_A_rec, BP_B_rec, P_K_rec] = xiK_to_matrices(xi_bar, n, m);
   
   eta=1;
   KcurrentGNL=KcurrentGNL-factorGNM*eta*(KcurrentGNL+inv(R+BP_B_rec)*(BP_A_rec'));
%% GNLS2
r=[X_t_reshaped;U_t_reshaped];
hatAB = inv(r*r')*(r*X_t1_reshaped');
Ahat=hatAB(1:3,:);
Bhat=hatAB(4:6,:);
khistoryGNL2(:,:,i)=KcurrentGNL2;
   phistoryGNL2(:,:,i)=dlyap((A+B*KcurrentGNL2)',Q+KcurrentGNL2'*R*KcurrentGNL2); 
   chistoryGNL2(i,j)=trace(phistoryGNL2(:,:,i));
Ptemp=dlyap((Ahat+Bhat*KcurrentGNL2)',Q+KcurrentGNL2'*R*KcurrentGNL2); 
KcurrentGNL2=KcurrentGNL2-factorGNM*eta*(KcurrentGNL2+inv(R+Bhat'*Ptemp*Bhat)*(Bhat'*Ptemp*A));
%% %% new
  khistoryGNL3(:,:,i)=KcurrentGNL3;
   phistoryGNL3(:,:,i)=dlyap((A+B*KcurrentGNL3)',Q+KcurrentGNL3'*R*KcurrentGNL3); 
   chistoryGNL3(i,j)=trace(phistoryGNL3(:,:,i));
   %[Hk, bk] = generate_Hb(X_t_reshaped, U_t_reshaped, X_t1_reshaped, KcurrentNPL2, Q, R, W);
   Ns=[8;14;26;50];
   %Ns=[21;27;39;62];
   D0=1;
    Ds2=[1,0.5,0.25,0.125];
 xi_bar=shrinking_multi_epoch_CSPD(X_t,U_t,X_t1,xi0, D0, numel(Ns), K, W, Q, R, opts,Ns,Ds2);
 [BP_A_rec, BP_B_rec, P_K_rec] = xiK_to_matrices(xi_bar, n, m);
   % NPG
   eta=1;
   KcurrentGNL3=KcurrentGNL3-factorGNM*eta*(KcurrentGNL3+inv(R+BP_B_rec)*(BP_A_rec'));
%% NPG CPD 
   khistoryNP(:,:,i)=KcurrentNP;
   phistoryNP(:,:,i)=dlyap((A+B*KcurrentNP)',Q+KcurrentNP'*R*KcurrentNP); 
   chistoryNP(i,j)=trace(phistoryNP(:,:,i));
   [xi_bar, y_bar, history] = conditional_stochastic_primal_dual_ball(KcurrentNP, X_t, U_t, X_t1, xi0, y0, params, r_xi, r_y, W, Q, R, opts);
   [BP_A_rec, BP_B_rec, P_K_rec] = xiK_to_matrices(xi_bar, n, m);
 
    KcurrentNP=KcurrentNP-0.05*((R+BP_B_rec)*KcurrentNP+BP_A_rec');
  
%% NPG LS
    khistoryNPL(:,:,i)=KcurrentNPL;
   phistoryNPL(:,:,i)=dlyap((A+B*KcurrentNPL)',Q+KcurrentNPL'*R*KcurrentNPL); 
   chistoryNPL(i,j)=trace(phistoryNPL(:,:,i));
   [Hk, bk] = generate_Hb(X_t_reshaped, U_t_reshaped, X_t1_reshaped, KcurrentNPL, Q, R, W);
     xi_bar=Hk \ bk;
     [BP_A_rec, BP_B_rec, P_K_rec] = xiK_to_matrices(xi_bar, n, m);
    KcurrentNPL=KcurrentNPL-0.05*((R+BP_B_rec)*KcurrentNPL+BP_A_rec');
 %% new
  khistoryNPL2(:,:,i)=KcurrentNPL2;
   phistoryNPL2(:,:,i)=dlyap((A+B*KcurrentNPL2)',Q+KcurrentNPL2'*R*KcurrentNPL2); 
   chistoryNPL2(i,j)=trace(phistoryNPL2(:,:,i));
   Ns=[8;14;26;51];
   D0=1;
    Ds2=[1,0.5,0.25,0.125];
 xi_bar=shrinking_multi_epoch_CSPD(X_t,U_t,X_t1,xi0, D0, numel(Ns), K, W, Q, R, opts,Ns,Ds2);
 [BP_A_rec, BP_B_rec, P_K_rec] = xiK_to_matrices(xi_bar, n, m);
   % NPG
    eta=1/norm(R+B'*phistoryNPL2(:,:,1)*B);
    KcurrentNPL2=KcurrentNPL2-0.05*((R+BP_B_rec)*KcurrentNPL2+BP_A_rec');
end
end

save("Data.mat")
function W = compute_W(Sigma_w)
    % Sigma_w: covariance matrix (n_x x n_x)

    [V, D] = eig(Sigma_w);      % eigen-decomposition
    lambda = diag(D);           % eigenvalues

    n_x = length(lambda);

    % IMPORTANT: W 的维度由 vecv() 决定
    % 因此先用第一个 term 来初始化维度
    first_term = vecv( sqrt(lambda(1)) * V(:,1) );
    W = zeros(size(first_term));

    % accumulate the sum
    W = W + first_term;

    for k = 2:n_x
        term = vecv( sqrt(lambda(k)) * V(:,k) );
        W = W + term;
    end
end

function out = vecv(v)
    n = length(v);
    out = zeros(n*(n+1)/2, 1);
    idx = 1;
    for i = 1:n
        for j = i:n
            out(idx) = v(i) * v(j);
            idx = idx + 1;
        end
    end
end

function s = vecs(P)
    n = size(P,1);
    s = zeros(n*(n+1)/2,1);
    idx = 1;

    for i = 1:n
        for j = i:n
            if i == j
                s(idx) = P(i,j);
            else
                s(idx) = 2 * P(i,j);
            end
            idx = idx + 1;
        end
    end
end

function P = unvecs(s)
m = length(s);
    n = (-1 + sqrt(1 + 8*m)) / 2;

    if abs(n - round(n)) > 1e-12
        error('Input length %d is not valid for vecs (n(n+1)/2).', m);
    end

    n = round(n);
    P = zeros(n,n);

    idx = 1;
    for i = 1:n
        for j = i:n
            if i == j
                val = s(idx);
            else
                val = s(idx) / 2;
            end
            P(i,j) = val;
            P(j,i) = val;
            idx = idx + 1;
        end
    end
end

function Gamma_hat = compute_Gamma_hat(x, u, K, x_next, W)
% compute_Gamma_hat  Computes the Gamma_t vector
%
% Inputs:
%   x       - x_t (n×1)
%   u       - u_t (m×1)
%   K       - feedback matrix (m×n)
%   x_next  - x_{t+1} (n×1)
%   W       - same dimension as vecv(x)
%
% Output:
%   Gamma_hat = stacked vector (size: n*m + m(m+1)/2 + n(n+1)/2)

    x = x(:);
    u = u(:);
    x_next = x_next(:);

    % ---------------------------
    % Part 1: 2 * x ⊗ (u - Kx)
    % ---------------------------
    term1 = 2 * kron(x, (u - K*x));

    % ---------------------------
    % Part 2: vecv(u) - vecv(Kx)
    % ---------------------------
    term2 = vecv(u) - vecv(K*x);

    % ---------------------------
    % Part 3: vecv(x) + W - vecv(x_next)
    % ---------------------------
    term3 = vecv(x) + W - vecv(x_next);

    % Stack all components
    Gamma_hat = [term1; term2; term3];
end

function c_hat = compute_c_hat(x, Q, R, K)

% Computes c_t = x' (Q + K' R K) x

    x = x(:);          % ensure column vector
    M = Q + K' * R * K;  % precompute the quadratic matrix
    c_hat = x' * M * x;     % scalar
end

function [Hk, bk] = generate_Hb(X_t, U_t, X_t1, K, Q, R, W)

% Inputs:
%   X_t  : n x N
%   U_t  : m x N
%   X_t1 : n x N
%   K    : m x n
%   Q,R  : cost matrices
%   W    : vector same size as vecv(x)
%
% Outputs:
%   Hk : N x dim(Gamma) regression matrix
%   bk : N x 1 vector

    [n, N] = size(X_t);
    m = size(U_t,1);

    % precompute vecv dimension
    dummy = vecv(X_t(:,1));
    dim_vecv = length(dummy);

    % compute Gamma dimension
    dim_Gamma = n*m + length(vecv(U_t(:,1))) + dim_vecv;  % from Gamma_hat structure

    % allocate
    Hk = zeros(N, dim_Gamma);
    bk = zeros(N,1);

    for j = 1:N
        x = X_t(:,j);
        u = U_t(:,j);
        x_next = X_t1(:,j);

        % compute Gamma_hat
        Gamma_t = compute_Gamma_hat(x, u, K, x_next, W);

        % store row in Hk
        Hk(j,:) = Gamma_t';  % transpose because row

        % compute c_t
        c_t = compute_c_hat(x, Q, R, K);
        bk(j) = c_t;
    end
end

function [BP_A, BP_B, P_K] = xiK_to_matrices(xi_K, n, m)
% xiK_to_matrices_fixed 从 xi_K 恢复矩阵
%
% Inputs:
%   xi_K : column vector, xi_K = [vec(B'PA); vecs(B'PB); vecs(P_K)]
%   n    : 状态维度 (size of P_K, also size of A)
%   m    : 控制输入维度 (columns of B)
%
% Outputs:
%   BP_A : n x m, B' * P_K * A
%   BP_B : m x m, B' * P_K * B
%   P_K  : n x n, P_K
%
% Note: 需要提前定义 unvecs.m (vecs 的逆操作)

    % 计算每部分长度
    len_BP_A = n * m;
    len_BP_B = m * (m + 1) / 2;
    len_P    = n * (n + 1) / 2;

    % 分割 xi_K
    xi1 = xi_K(1:len_BP_A);                     % vec(B'PA)
    xi2 = xi_K(len_BP_A+1:len_BP_A+len_BP_B);  % vecs(B'PB)
    xi3 = xi_K(end-len_P+1:end);               % vecs(P_K)

    % 恢复矩阵
    BP_A = reshape(xi1, m, n)';  % n x m
    BP_B = unvecs(xi2);          % m x m 对称
    P_K  = unvecs(xi3);          % n x n 对称
end

function [xi_bar, y_bar, history] = conditional_stochastic_primal_dual_ball( K, X_cell, U_cell, X1_cell, xi0, y0, params, r_xi, r_y, W, Q, R, opts)
% Closed-form projections onto Euclidean balls for both y and xi.
% Inputs:
%   K, X_cell, U_cell, X1_cell - data and policy (cells length >= E)
%   xi0, y0  - initial vectors
%   params   - struct with fields E, eta, lambda, zeta (scalars or vectors)
%   r_xi     - radius for xi-ball constraint
%   r_y      - radius for y-ball constraint
%   W, Q, R  - auxiliary params used in generate_Hb / Gamma / c_t
%   opts     - options, supports opts.verbose
% Outputs:
%   xi_bar, y_bar - weighted averages as in algorithm
%   history       - stored iterates
    if nargin < 13, opts = struct(); end
    if ~isfield(opts,'verbose'), opts.verbose = false; end
    E = params.E;

 %   % sequences
 %   if isscalar(params.eta),    eta_seq = repmat(params.eta, E,1); else eta_seq = params.eta(:); end
 %   if isscalar(params.lambda), lambda_seq = repmat(params.lambda, E,1); else lambda_seq = params.lambda(:); end
 %   if isscalar(params.zeta),   zeta_seq = repmat(params.zeta, E,1); else zeta_seq = params.zeta(:); end
    eta_seq    = expand_stepsize(params.eta,    E);
    lambda_seq = expand_stepsize(params.lambda, E);
    zeta_seq   = expand_stepsize(params.zeta,   E);
    % init
    xi_prev = xi0(:);
    xi_prev2 = xi0(:);
    y_prev = y0(:);

    xi_dim = length(xi_prev);
    y_dim  = length(y_prev);

    xi_hist = zeros(xi_dim, E);
    y_hist  = zeros(y_dim, E);

    for e = 1:E
        % extrapolation
        G_e = xi_prev + zeta_seq(e) * (xi_prev - xi_prev2);

        % construct H_e and b_e from data of epoch e
        r = randi(E);
        X_t = X_cell(:,:,r); 
        U_t = U_cell(:,:,r); 
        X1_t = X1_cell(:,:,r);
        [H_e, b_e] = generate_Hb(X_t, U_t, X1_t, K, Q, R,W);   % uses compute_Gamma_hat & compute_c_t
        b_e = b_e(:);

        % y-update: closed-form + projection onto ball radius r_y
        f = - H_e * G_e + b_e;
        lambda_e = lambda_seq(e);
        y_unc = y_prev - (1/lambda_e) * f;
        y_new = proj_ball(y_unc, r_y);

        % xi-update: closed-form + projection onto ball radius r_xi
        eta_e = eta_seq(e);
        grad_lin = H_e' * y_new;
        xi_unc = xi_prev - (1/eta_e) * grad_lin;
        xi_new = proj_ball(xi_unc, r_xi);

        % store
        xi_hist(:, e) = xi_new;
        y_hist(:, e)  = y_new;

       % if opts.verbose
       %     fprintf('Epoch %d: ||xi diff||=%.3e ||y diff||=%.3e\n', e, norm(xi_new-xi_prev), norm(y_new-y_prev));
       % end

        % shift
        xi_prev2 = xi_prev;
        xi_prev = xi_new;
        y_prev = y_new;
    end

    % weighted average
    weights = (1:E)';
    Wt = 2/(E*(E+1));
    xi_bar = Wt * (xi_hist * weights);
    y_bar  = Wt * (y_hist  * weights);

    history.xi = xi_hist;
    history.y  = y_hist;
end

function z = proj_ball(x, r)
    if r <= 0
        z = zeros(size(x));
        return;
    end
    nrm = norm(x);
    if nrm <= r
        z = x;
    else
        z = (r / nrm) * x;
    end

end

function seq = expand_stepsize(val, E)
    % If val is scalar → expand to length E
    if isscalar(val)
        seq = val * ones(E,1);
    else
        val = val(:);
        if length(val) ~= E
            error("Step size vector length (%d) must equal E (%d).", length(val), E);
        end
        seq = val;
    end
end

function [X_t,U_t,X_t1]=datacollection(K,N,m,n,Sigma_x,Sigma_u,Sigma_w,A,B)
X_t  = zeros(n, N,K);
U_t  = zeros(m, N,K);
X_t1 = zeros(n, N,K);



for k = 1:K
    x_t = zeros(n, N);
    u_t = zeros(m, N);
    x_t1 = zeros(n, N);

    for j = 1:N
        % input output 
        x_t(:,j) = mvnrnd(zeros(n,1), Sigma_x)';
        u_t(:,j) = mvnrnd(zeros(m,1), Sigma_u)';

        % x_{t+1} = A x_t + B u_t + w_t
        w_t = mvnrnd(zeros(n,1), Sigma_w)';
        x_t1(:,j) = A * x_t(:,j) + B * u_t(:,j) + w_t;
    end

    % 保存结果
    X_t(:,:,k) = x_t;
    U_t(:,:,k) = u_t;
    X_t1(:,:,k)  = x_t1;
end
end

function xi_hat = shrinking_multi_epoch_CSPD(X_cell,U_cell,X1_cell,xi0, D0, S, K, W, Q, R, opts,Ns,Ds2)
% Inputs:
%   X      : 原始可行集合（这里只是radius，用投影替代）
%   xi0    : 初始  \tilde{\xi}_0
%   D0     : 初始半径 D0
%   S      : epoch 数
%   K      : 用于构造 H_k 和 b_k 的 policy
%   data   : 数据 {x_t, u_t, x_{t+1}} 结构体或 cell
%
% Output:
%   xi_hat : \tilde{\xi}_S

    xi_prev = xi0;
    index=1;
   
    for s = 1:S
        center = xi_prev;
        radius = Ds2(s);   % 你的约束是 Euclidean ball
        e_vec = 1:Ns(s);
        eta = 0.001*sqrt(e_vec);        % η_e = 1/e
        lambda =0.001*sqrt(e_vec);
        zeta   = (e_vec - 1) ./ e_vec;
        E      = Ns(s);
        r_y=1;
        y0=0;
        xi_s = conditional_stochastic_primal_dual_ball2( K, X_cell(:,:,index:index+Ns(s)), U_cell(:,:,index:index+Ns(s)), X1_cell(:,:,index:index+Ns(s)), xi_prev, y0, eta,lambda,zeta,E, radius, zeros(21,1),r_y, W, Q, R);
        xi_prev = xi_s;
        index=index+Ns(s);
    end

    %% Final output
    xi_hat = xi_prev;

end

function [xi_bar, y_bar, history] = conditional_stochastic_primal_dual_ball2( K, X_cell, U_cell, X1_cell, xi0, y0, eta,lambda,zeta,E, r_xi,center, r_y, W, Q, R)
% Closed-form projections onto Euclidean balls for both y and xi.
% Inputs:
%   K, X_cell, U_cell, X1_cell - data and policy (cells length >= E)
%   xi0, y0  - initial vectors
%   params   - struct with fields E, eta, lambda, zeta (scalars or vectors)
%   r_xi     - radius for xi-ball constraint
%   r_y      - radius for y-ball constraint
%   W, Q, R  - auxiliary params used in generate_Hb / Gamma / c_t
%   opts     - options, supports opts.verbose
% Outputs:
%   xi_bar, y_bar - weighted averages as in algorithm
%   history       - stored iterates
    %if nargin < 13, opts = struct(); end
    %if ~isfield(opts,'verbose'), opts.verbose = false; end
    %E = params.E;

 %   % sequences
 %   if isscalar(params.eta),    eta_seq = repmat(params.eta, E,1); else eta_seq = params.eta(:); end
 %   if isscalar(params.lambda), lambda_seq = repmat(params.lambda, E,1); else lambda_seq = params.lambda(:); end
 %   if isscalar(params.zeta),   zeta_seq = repmat(params.zeta, E,1); else zeta_seq = params.zeta(:); end
    eta_seq    = eta;
    lambda_seq = lambda;
    zeta_seq   = zeta;
    % init
    xi_prev = xi0(:);
    xi_prev2 = xi0(:);
    y_prev = y0(:);

    xi_dim = length(xi_prev);
    y_dim  = length(y_prev);

    xi_hist = zeros(xi_dim, E);
    y_hist  = zeros(y_dim, E);

    for e = 1:E
        % extrapolation
        G_e = xi_prev + zeta_seq(e) * (xi_prev - xi_prev2);

        % construct H_e and b_e from data of epoch e
        r = randi(E);
        X_t = X_cell(:,:,r); 
        U_t = U_cell(:,:,r); 
        X1_t = X1_cell(:,:,r);
        [H_e, b_e] = generate_Hb(X_t, U_t, X1_t, K, Q, R,W);   % uses compute_Gamma_hat & compute_c_t
        b_e = b_e(:);

        % y-update: closed-form + projection onto ball radius r_y
        f = - H_e * G_e + b_e;
        lambda_e = lambda_seq(e);
        y_unc = y_prev - (1/lambda_e) * f;
        y_new = proj_ball(y_unc, r_y);

        % xi-update: closed-form + projection onto ball radius r_xi
        eta_e = eta_seq(e);
        grad_lin = H_e' * y_new;
        xi_unc = xi_prev - (1/eta_e) * grad_lin;
        xi_new = proj_ball_center(xi_unc,center, r_xi);

        % store
        xi_hist(:, e) = xi_new;
        y_hist(:, e)  = y_new;

       % if opts.verbose
       %     fprintf('Epoch %d: ||xi diff||=%.3e ||y diff||=%.3e\n', e, norm(xi_new-xi_prev), norm(y_new-y_prev));
       % end

        % shift
        xi_prev2 = xi_prev;
        xi_prev = xi_new;
        y_prev = y_new;
    end

    % weighted average
    weights = (1:E)';
    Wt = 2/(E*(E+1));
    xi_bar = Wt * (xi_hist * weights);
    y_bar  = Wt * (y_hist  * weights);

    history.xi = xi_hist;
    history.y  = y_hist;
end



function z = proj_ball_center(x, center, r)
% Projection of x onto ball {z: ||z - center|| <= r}
    if r <= 0
        z = center;
        return;
    end
    diff = x - center;
    nrm = norm(diff);
    if nrm <= r
        z = x;
    else
        z = center + (r/nrm) * diff;
    end
end