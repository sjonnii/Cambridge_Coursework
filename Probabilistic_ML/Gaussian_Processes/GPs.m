%A
clear all;
clc;
load('cw1a.mat')
x = x; y=y;
xs = linspace(-3, 3, 300)';

meanfunc = [];                    
covfunc = @covSEiso;              
likfunc = @likGauss;              

hyp = struct('mean', [], 'cov', [-1 0], 'lik', 0);

hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
hold on; plot(xs, mu); plot(x, y, '+')
%%
%A
clear all;
clc;
load('cw1a.mat')
x = x; y=y;
xs = linspace(-3, 3, 300)';

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [2.08 -0.36], 'lik', 0);

%hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
hold on; plot(xs, mu); plot(x, y, '+')

%%
%B

N = linspace(-2,2,5)
M = linspace(-2,2,5)
K = linspace(-2,2,5)

likeli = zeros(5,5,5);
cov1 = zeros(5,5,5);
cov2 = zeros(5,5,5);
total_like = zeros(5,5,5);

for n = 1:5
    for m = 1:5
        for k = 1:5
        hyp = struct('mean', [], 'cov', [N(n), M(m)], 'lik', K(k));    
        hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);        
        likeli(n,m,k) = hyp2.lik;
        cov1(n,m,k) = hyp2.cov(1);
        cov2(n,m,k) = hyp2.cov(2);      
        total_like(n,m,k) = gp(hyp2,@infGaussLik, [], covfunc, likfunc, x, y)
        end
        
    end
end

[M, I] = min(total_like(:));
[ind1, ind2, ind3] = ind2sub(size(total_like), I);

%% 
%C
clear all;
clc;
load('cw1a.mat')
xs = linspace(-3, 3, 300)';

meanfunc = [];                    
covfunc = @covPeriodic;         
likfunc = @likGauss;              

hyp = struct('mean', [], 'cov', [1, 1, 1], 'lik', 0)

hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y)
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
hold on; plot(xs, mu); plot(x, y, '+')
%%
%D
clc; clear;
meanfunc = [];
covfunc = {@covProd, {@covPeriodic, @covSEiso}};
likfunc = @likGauss;
hyp = struct('mean', [], 'cov', [-0.5, 0, 0, 2, 0], 'lik', 0);

n = 200;
x = linspace(-5,5,n)';
K = feval(covfunc{:}, hyp.cov, x);
diag = 1e-6*eye(200);
K = K + diag;
chol = chol(K);
y = chol' * gpml_randn(2,n,3)
plot(x, y)
hold on;
%%
%E a)
clc; clear;
load('cw1e.mat')
%mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11));
[t1 t2] = meshgrid(-10:0.1:10,-10:0.1:10);
z1 = [reshape(t1,[],1), reshape(t2, [],1)];
meanfunc = [];                    
covfunc = @covSEard;              
likfunc = @likGauss;              
hyp = struct('mean', [], 'cov', [0,1,0], 'lik', 0);
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y)
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, z1);

mesh(t1,t2,reshape(mu,201,201))

%%
%E b)
clc; clear;
load('cw1e.mat')
%mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11));
[t1 t2] = meshgrid(-10:0.1:10,-10:0.1:10);
z1 = [reshape(t1,[],1), reshape(t2, [],1)];

meanfunc = [];                    
covfunc = {@covSum, {@covSEard, @covSEard}};              
likfunc = @likGauss;              

hyp = struct('mean', [], 'cov',  0.1*randn(6,1), 'lik', 0);

hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y)
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, z1);

mesh(t1,t2,reshape(mu,201,201))




