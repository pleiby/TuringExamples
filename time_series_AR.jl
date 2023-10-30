# source: TuringLang/TuringExamples repository

#Import Turing, Distributions, StatsBase, DataFrames and CSV
using Turing, Distributions, StatsBase, DataFrames, CSV

# Import MCMCChain, Plots and StatsPlots for visualizations and diagnostics
using MCMCChains, Plots, StatsPlots

# Set a seed for reproducibility.
using Random
Random.seed!(12);

# Turn off progress monitor.
Turing.turnprogress(false)

# Load in the shampoo dataset (can be downloaded from https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv)
println("Loading the dataset")
df = CSV.read("./data/shampoo.csv", DataFrame)
s = df.Sales
# pyplot() # Choose the pyplot (matplotlib) backend (Or another backend)
plot(s, reuse=false, title="Shampoo dataset")
gui()

println("Split into training and test sets.")
train_percentage = 0.9
s_train = s[1:floor(Int, train_percentage * length(s))]
N = length(s_train)
println("We will predict for the next $(length(s)-N) obs using the data from the past $N obs.")

println("Plot the training data")
plot(s_train, reuse=false, title="Train Data")
# gui()

println("Plotting ACF and PACF plots")
s1 = scatter([1, 2, 3, 4, 5], autocor(s, [1, 2, 3, 4, 5]), title="ACF", ylim=[0.3, 0.8])
s2 = scatter([1, 2, 3, 4, 5], pacf(s, [1, 2, 3, 4, 5]), title="PACF", ylim=[0.3, 0.8])
plot(s1, s2, layout=(2, 1), reuse=false)
# gui()
println("The PACF plot cuts off at k = 2, so we will have an AR(2) model for this dataset")

#Defining the model
σ = 1.0
@model AR(x, N) = begin
    # prior
    α ~ Normal(0, σ)
    beta_1 ~ Uniform(-1, 1)
    beta_2 ~ Uniform(-1, 1)
    sigma ~ Exponential(1) # ??? Unused
    # likelihood
    for t in 3:N
        μ = α + beta_1 * x[t-1] + beta_2 * x[t-2]
        x[t] ~ Normal(μ, 0.1)
    end
end;

# Sample using NUTS(n_iters::Int, n_adapts::Int, δ::Float64), where:
# n_iters::Int : The number of samples to pull.
# n_adapts::Int : The number of samples to use with adapatation.
# δ::Float64 : Target acceptance rate.

# chain = sample(AR(s_train, N), NUTS(5000, 200, 0.65))
model = AR(s_train, N)
samples = 5_000
sampler = NUTS(200, 0.65)
chain = sample(model, sampler, samples)

println("Chain has been sampled; Now let us visualise it!")
plot(chain, reuse=false, title="Sampler Plot")
# gui()

#Plotting the corner plot for the chain
corner(chain, reuse=false, title="Corner Plot")
# gui()

println("Removing the warmup samples...")
chains_new = chain[50:4800]
show(chains_new)

# Getting the mean values of the sampled parameters
#v beta_1 = mean(chains_new[:beta_1].value) # returns ERROR: type AxisArray has no field valeu
beta_1 = mean(chains_new[:beta_1][:, 1]) # returns ERROR: type AxisArray has no field valeu
beta_2 = mean(chains_new[:beta_2][:, 1])


println("Obtaining the test data")
s_test = s[N+1:length(s)]

println("Obtaining the predicted results using the mean values of beta_1 and beta_2")
# ??? Three strange things here:
#  prediction using lagged actual (vs fitted) values; exog normal disturbance term; dropped α from model
s_pred = Float64[]
first_ele = s_train[N] * beta_1 + s_train[N-1] * beta_2 + rand(Normal(0, 1))
push!(s_pred, first_ele)
second_ele = s_pred[1] * beta_1 + s_train[N] * beta_2 + rand(Normal(0, 1))
push!(s_pred, second_ele)
for i = 3:length(s_test)
    next_ele = s_pred[i-1] * beta_1 + s_pred[i-2] * beta_2 + rand(Normal(0, 1))
    push!(s_pred, next_ele)
end

println("Plotting the test and the predicted data for comparison")
p_ests = plot(s_test, reuse=false, title="Predicted vs Test Comparison", label="Test")
plot!(p_ests, s_pred, label="Predicted")
# gui()

# println("Press ENTER to exit")
# read(stdin, Char)