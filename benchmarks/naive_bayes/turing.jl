using Random: seed!
seed!(1)

include("data.jl")

data = get_data()

using LazyArrays, Memoization, Turing

lazyarray(f, x...) = LazyArray(Base.broadcasted(f, x...))
@model naive_bayes(image, label, D, N, C, ::Type{T}=Float64) where {T<:Real} = begin
    m ~ filldist(Normal(0, 10), D, C)
    image ~ arraydist(lazyarray(Normal, m[:,label], 1))
end 

N = 10
D = data["D"]
model = naive_bayes(data["image"][1:D, 1:N], data["label"][1:N], data["D"], data["N"], data["C"])

step_size = 0.1
n_steps = 4

test_zygote = false
test_tracker = false

include("../infer_turing.jl")

# Save result

if !isnothing(chain)
    using BSON

    m_data = chain[:m].value.data

    m_bayes = mean(
        map(
            i -> reconstruct(pca, Matrix{Float64}(reshape(m_data[i,:,1], D_pca, 10))), 
            1_000:100:2_000
        )
    )

    bson("result.bson", m_bayes=m_bayes)
end