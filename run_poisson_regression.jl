using CSV
using PyPlot


include("poisson_regression.jl")


df = CSV.read("data3a.csv")

x = df[:x]
y = vec(df[:y])

x = x - mean(x)
x = x ./ std(x)

X = reshape(x, 1, size(x, 1))


model = PoissonRegression(learning_rate = 1.2e-2)

model = fit!(model, X, y, 800)

x_min, x_max = minimum(x), maximum(x)
x_test = x_min-0.1:0.01:x_max+0.1
X_test = reshape(x_test, 1, length(x_test))
y_pred = predict(model, X_test)


xlabel("x")
ylabel("y")
scatter(x, y, color="blue")
plot(x_test, y_pred, color="red", label="expected value")

legend()
show()
