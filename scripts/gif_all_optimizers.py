import copy

from gradient_free_optimizers import (
    HillClimbingOptimizer,
    StochasticHillClimbingOptimizer,
    RepulsingHillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    DownhillSimplexOptimizer,
    RandomSearchOptimizer,
    PowellsMethod,
    GridSearchOptimizer,
    RandomRestartHillClimbingOptimizer,
    RandomAnnealingOptimizer,
    PatternSearch,
    ParallelTemperingOptimizer,
    ParallelAnnealingOptimizer,
    ParticleSwarmOptimizer,
    EvolutionStrategyOptimizer,
    BayesianOptimizer,
    TreeStructuredParzenEstimators,
    ForestOptimizer,
    EnsembleOptimizer,
)

from surfaces.test_functions import (
    SphereFunction,
    AckleyFunction,
    BealeFunction,
    MatyasFunction,
)

from gradient_free_optimization_plots import search_path_gif


n_iter_s = 250
n_iter_m = 250
n_iter_l = 25

initialize_s = {"random": 10}
initialize_m = {"vertices": 4, "grid": 4, "random": 2}
initialize_l = {"vertices": 4, "random": 6}

opt_list = [
    (HillClimbingOptimizer, n_iter_s, initialize_s),
    (StochasticHillClimbingOptimizer, n_iter_s, initialize_s),
    (RepulsingHillClimbingOptimizer, n_iter_s, initialize_s),
    (SimulatedAnnealingOptimizer, n_iter_s, initialize_s),
    (DownhillSimplexOptimizer, n_iter_s, initialize_s),
    (RandomSearchOptimizer, n_iter_s, initialize_s),
    # (PowellsMethod, n_iter_s, initialize_s),
    (GridSearchOptimizer, n_iter_s, initialize_s),
    (RandomRestartHillClimbingOptimizer, n_iter_s, initialize_s),
    (RandomAnnealingOptimizer, n_iter_s, initialize_s),
    (PatternSearch, n_iter_s, initialize_s),
    (ParallelTemperingOptimizer, n_iter_m, initialize_m),
    (ParallelAnnealingOptimizer, n_iter_m, initialize_m),
    (ParticleSwarmOptimizer, n_iter_m, initialize_m),
    (EvolutionStrategyOptimizer, n_iter_m, initialize_m),
    (BayesianOptimizer, n_iter_l, initialize_l),
    (TreeStructuredParzenEstimators, n_iter_l, initialize_l),
    (ForestOptimizer, n_iter_l, initialize_l),
    (EnsembleOptimizer, n_iter_l, initialize_l),
]

opt_list = [
    (BayesianOptimizer, n_iter_l, initialize_l),
]

parameter_template = {
    "path": "../gifs",
    "opt_para": {},
    "random_state": 1,
}


sphere_function = SphereFunction(n_dim=2, metric="score")
ackley_function = AckleyFunction(metric="score")
beale_function = BealeFunction(metric="score")
matyas_function = MatyasFunction(metric="score")


obj_func_l = [
    sphere_function,
    ackley_function,
    # beale_function,
    # matyas_function,
]


para_d = {}
for opt_ in opt_list:
    optimizer = opt_[0]

    opt_name = optimizer.name
    n_iter = opt_[1]
    initialize = opt_[2]

    para_d[opt_name] = copy.deepcopy(parameter_template)
    para_d[opt_name]["optimizer"] = optimizer
    para_d[opt_name]["n_iter"] = n_iter
    para_d[opt_name]["initialize"] = initialize

    for obj_func_ in obj_func_l:
        setup_name = opt_name + "___" + obj_func_.__name__ + "_.gif"
        setup_name = "_".join(setup_name.split())
        print("setup_name", setup_name)

        para_d[opt_name]["name"] = setup_name
        para_d[opt_name]["objective_function"] = obj_func_
        para_d[opt_name]["search_space"] = obj_func_.search_space(min=-3.5, max=6)

        search_path_gif(**para_d[opt_name])
