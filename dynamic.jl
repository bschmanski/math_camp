using LinearAlgebra, Plots

alpha = 0.5
beta = 2/3

n = 100
n_iter = 50

k = collect(range(0.1, 10, n))
V_t = zeros(n, 1)

# V(K) = max log(k^alpha - ktilde) + beta*V(ktilde)
#           ktilde in [0, k^alpha]

V(n, k_tilde) = log(k[n]^alpha - k_tilde) + beta*V_t[n]

for s in range(1, n_iter)
    for t in range(1, n)
        k_tildes = collect(range(0.1, k[t]^alpha, n))
        V_t[t] = findmax(V.(t, k_tildes))[1]
    end
end

plot(k, V_t)