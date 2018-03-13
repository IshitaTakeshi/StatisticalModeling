const AA = AbstractArray


srand(1234)


type PoissonRegression
    θ::AA{Float64, 1}
    α::Float64
    η::Float64
    function PoissonRegression(;learning_rate = 1e-4)
        return new(zeros(0), 0.0, learning_rate)
    end
end


function Δθ(xᵢ::AA{Float64, 1}, yᵢ::Int, θ::AA{Float64, 1}, α::Float64)
    xᵢ * exp(dot(θ, xᵢ) + α) - yᵢ * xᵢ
end


function Δθ(X::AA{Float64, 2}, y::AA{Int, 1},
            θ::AA{Float64, 1}, α::Float64)

    Δ = zeros(Float64, size(X, 1))
    for (i, yᵢ) in enumerate(y)
        xᵢ = view(X, :, i)
        Δ += xᵢ * exp(dot(θ, xᵢ) + α) - yᵢ * xᵢ
    end
    return Δ
end


Δα(yᵢ::Int, α::Float64) = exp(α) - yᵢ


Δα(y::AA{Int64, 1}, α::Float64) = sum(map(yᵢ -> exp(α) - yᵢ, y))


function init_params!(model::PoissonRegression, X::AA{Float64, 2})
    model.θ = zeros(size(X, 1))  # 0.001 * randn(size(X, 1))
    model.α = 0  # 0.001 * randn()  # genere a scalar value
    return model
end


function fit!(model::PoissonRegression, X::AA{Float64, 2}, y::AA{Int, 1},
              n_iter::Int)
    init_params!(model, X)

    println(log_likelihood(model, X, y))
    for i in 1:n_iter
        update!(model, X, y)
        println(model)
        println(log_likelihood(model, X, y))
    end
    return model
end


function predict(model::PoissonRegression, X::AA{Float64, 2})
    θ, α = model.θ, model.α

    N = size(X, 2)
    y = zeros(N)

    for i in 1:N
        xᵢ = view(X, :, i)
        y[i] = exp(dot(θ, xᵢ) + α)
    end

    return y
end


function log_likelihood(model::PoissonRegression,
                        X::AA{Float64, 2}, y::AA{Int, 1})
    θ, α = model.θ, model.α

    L = 0
    for (i, yᵢ) in enumerate(y)
        xᵢ = view(X, :, i)
        t = dot(θ, xᵢ) + α
        s = sum(log(i) for i in 1:yᵢ)  # equivalent to log(factorial(yᵢ))
        L += yᵢ * t - exp(t) - s
    end
    return L
end


function update!(model::PoissonRegression, X::AA{Float64, 2}, y::AA{Int, 1})
    θ, α, η = model.θ, model.α, model.η

    for (i, yᵢ) in enumerate(y)
        xᵢ = view(X, :, i)
        model.θ = θ - η * Δθ(xᵢ, yᵢ, θ, α)
        model.α = α - η * Δα(yᵢ, α)
    end
    return model
end
