
function multipliers_gradients!(mults, zonation, K_grads, K_guess, mults_grads)
    for zone in 1:size(mults)[1]
        mults_grads[zone] = K_grads[zonation .== zone]'*K_guess[:][zonation .== zone]
    end
end

function case_gradients(mults, zonation, K_guess, model, loss)
    prm = Dict(
        "perm" => K_guess[:].*mults[zonation],
    )

    mults_grads = ones(Float64, size(mults))
    opt = setup_reservoir_dict_optimization(prm, model)
    free_optimization_parameter!(opt, "perm", rel_min = 0.1, rel_max = 10.0)
    loss, K_grads = parameters_gradient_reservoir(opt, loss, raw_output=true)
    multipliers_gradients!(mults, zonation, K_grads, K_guess, mults_grads)
    return loss, K_grads, mults_grads
end