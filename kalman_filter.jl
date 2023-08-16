using LinearAlgebra, Distributions, Plots

N = 100

# Initialization
F = [1 0; .1 1] #A
G = [1; 0]      #C
H = [1 1]       #G
Q = 1
R = 5

x = zeros(N+1, 2)
y = zeros(N)

w = rand(Normal(0, sqrt(Q)), N)
v = rand(Normal(0, sqrt(R)), N)

x[1,:] = rand(MvNormal([0,0], [20 5; 5 20]))

for t in range(1, N)
   x[t+1,:] = (F*x[t,:]) + (G*w[t])
   y[t,:] = (H*x[t+1,:]) .+ v[t]
end

x_hat = zeros(N+1, 2)
x_hat[1,:] = [0.5 0.5]
P_hat = [1 1; 1 1]

# Estimate x given y with Kalman Filter
function kalman_filter1()
    global P_hat
    for t in range(1, N)
        # Kalman prediction
        x_bar = F*x_hat[t,:]
        P_bar = (F*P_hat*F') + G*Q*G'
 
        # Kalman update
        S = H*P_bar*H' .+ R
        x_hat[t+1,:] = x_bar + P_bar*H'*inv(S)*(y[t,:] - H*x_bar)
        P_hat = P_bar - P_bar*H'*inv(S)*(P_bar*H')'
    end
end

kalman_filter1()
x_hat1 = x_hat

x_hat = zeros(N+1, 2)
x_hat[1,:] = [0.5 0.5]
P_hat = [1 1; 1 1]

# Other update rule
function kalman_filter2()
    global P_hat
    for t in range(1, N)
        # Kalman prediction
        x_bar = F*x_hat[t,:]
        P_bar = (F*P_hat*F') + G*Q*G'

        # Kalman update
        x_hat[t+1,:] = P_hat * (H'*inv(R)*y[t,:] + inv(P_bar)*x_bar)
        P_hat = inv(H'*inv(R)*H + inv(P_bar))
    end
end

kalman_filter2()
x_hat2 = x_hat

plot([x[:,1], x_hat1[:,1], x_hat2[:,1]], [x[:,2], x_hat1[:,2], x_hat2[:,2]], label=["real" "kalman1" "kalman2"])
