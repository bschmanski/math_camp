using LinearAlgebra, Optim, Distributions

n = 10000
x = [ones(n) randn(n) randn(n)]
b = [0.5, 1, 2]
y = x*b + randn(n);

# OLS
b_ols = inv(x'x) * x'y
println("OLS estimate: ", b_ols)

# GMM
gmm(b) = (1/n .* x'*(y - x*b))' * I * (1/n .* x'*(y - x*b))
b_gmm = Optim.minimizer(optimize(gmm, [0., 0., 0.]))
println("GMM estimate: ", b_gmm)

# MLE
mle(theta) = (-1/n)*(sum(log.(pdf.(Normal(0, 1), ((y - x*theta[1:3])/theta[4]))))-log(theta[4]))
b_mle = Optim.minimizer(optimize(mle, [1., 1., 1., 1.]))
println("MLE estimate: ", b_mle[1:3])
