from ConfigSpace import (
    ConfigurationSpace,
    EqualsCondition,
    InCondition,
    NotEqualsCondition,
)
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRFClassifier, XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


def get_default_cs(algo) -> ConfigurationSpace:
    if algo == HistGradientBoostingClassifier:
        return ConfigurationSpace(
            {
                "learning_rate": (0.01, 0.3),
                # "max_iter": (50, 200), # set by smac instead
                "max_depth": (3, 50),
                "min_samples_leaf": (1, 50),
                "max_leaf_nodes": (2, 50),
                "l2_regularization": (0.0, 0.3),
                "max_bins": (5, 255),
            }
        )
    elif algo == XGBRFClassifier or algo == XGBClassifier:
        return ConfigurationSpace(
            {
                "colsample_bynode": (0.001, 1.0),
                "learning_rate": (0.0, 1.0),
                "reg_lambda": (0.0, 100.0),  # TODO: Look more into this one
                "subsample": (0.001, 1.0),
                # "objective": "binary:logistic",
                # "use_label_encoder": None,
                # "base_score": None,
                # "booster": None,
                # "callbacks": None,
                "colsample_bylevel": (0.001, 1.0),
                "colsample_bytree": (0.001, 1.0),
                # "early_stopping_rounds": None,
                # "enable_categorical": False,
                # "eval_metric": None,
                # "feature_types": None,
                "gamma": (0, 20),
                # "gpu_id": None,
                # "grow_policy": None,
                # "importance_type": None,
                # "interaction_constraints": None,
                # "max_bin": None,
                # "max_cat_threshold": None,
                # "max_cat_to_onehot": None,
                "max_delta_step": (0, 10),
                "max_depth": (1, 12),
                # "max_leaves": None,
                "min_child_weight": (0, 10),
                # "missing": nan,
                # "monotone_constraints": None,
                # "n_estimators": 100,
                # "n_jobs": None,
                # "num_parallel_tree": None,
                # "predictor": None,
                # "random_state": None,
                "reg_alpha": (0.0, 100.0),  # TODO: Look more into this one
                # "sampling_method": None,
                # "scale_pos_weight": 1.88126,  # sum(negative instances) / sum(positive instances) = 1014 / 539
                # "tree_method": None,
                # "validate_parameters": None,
                # "verbosity": None,
            }
        )
    elif algo == CatBoostClassifier:
        cs = ConfigurationSpace(
            {
                "iterations": 400,
                ## "nan_mode": ["Min", "Max"],
                # "eval_metric": "Logloss", #TODO: Look more into objectives/metrics
                # "iterations": 1000, # set by SMAC
                ## "sampling_frequency": ["PerTree", "PerTreeLevel"],
                # "leaf_estimation_method": "Newton", #TODO
                ## "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
                ## "penalties_coefficient": (1.0, 10.0),
                ## "boosting_type": ["Ordered", "Plain"],
                ## "model_shrink_mode": ["Constant", "Decreasing"],
                ## "feature_border_type": [
                ##     "Median",
                ##     "Uniform",
                ##     "UniformAndQuantiles",
                ##     "MaxLogSum",
                ##     "MinEntropy",
                ##     "GreedyLogSum",
                ## ],
                # "bayesian_matrix_reg": 0.10000000149011612, #TODO: not in docs?
                # "eval_fraction": 0, #TODO:  not in docs?
                # "force_unit_auto_pair_weights": False, #TODO:  not in docs?
                "l2_leaf_reg": (1.0, 10.0),
                "random_strength": (0.0, 1.0),
                ## "rsm": (0.00001, 1.0),
                ## "boost_from_average": [True, False],
                # "model_size_reg": 0.5, #TODO:  not in docs?
                # "pool_metainfo_options": {"tags": {}}, #TODO:  not in docs?
                # "subsample": (0.0, 1.0), #FIXME This currently causes an error
                # "use_best_model": False,
                # "class_names": [0, 1], # auto determined
                # "random_seed": 0, # set by SMAC
                "depth": (4, 10),
                ## "posterior_sampling": [True, False],
                ## "border_count": (1, 256),
                # "classes_count": 0, #TODO:  not in docs?
                ## "auto_class_weights": ["None", "Balanced", "SqrtBalanced"],
                # "sparse_features_conflict_fraction": 0, #TODO:  not in docs?
                ## "leaf_estimation_backtracking": [
                ##     "No",
                ##     "AnyImprovement",
                ## ],  # only supported on GPU: , "Armijo"],
                # "best_model_min_trees": 1, #TODO
                ## "model_shrink_rate": (0.00001, 1.0),
                # "min_data_in_leaf": 1, #TODO
                # "loss_function": "Logloss", #TODO: Look more into objectives/metrics
                ## "learning_rate": (0.00001, 1.0),
                ## "score_function": [
                ##     "Cosine",
                ##     "L2",
                ## ],  # only supported on GPU:, "NewtonCosine", "NewtonL2"],
                # "task_type": "CPU", #N/A
                # "leaf_estimation_iterations": 10, #TODO
                ## "bootstrap_type": [
                ##     "Bayesian",
                ##     "Bernoulli",
                ##     "MVS",
                ##     "No",
                ## ],  # only supported on GPU: , "Poisson"]
                ## "max_leaves": (1, 64),
                "bagging_temperature": (0.0, 10.0),
            }
        )

        # max_leaves option works only with lossguide tree growing
        # cs.add_condition(
        #     EqualsCondition(cs["max_leaves"], cs["grow_policy"], "Lossguide")
        # )

        # subsample can only be used with Poisson, Bernoulli, MVS bootstrap type
        # FIXME See above; subsample currently causes an error
        # cs.add_condition(
        #     InCondition(
        #         cs["subsample"], cs["bootstrap_type"], ["Poisson", "Bernoulli", "MVS"]
        #     )
        # )

        # Posterior Sampling requires Сonstant Model Shrink Mode
        # cs.add_condition(
        #     EqualsCondition(
        #         cs["posterior_sampling"], cs["model_shrink_mode"], "Constant"
        #     )
        # )

        # Ordered boosting is not supported for nonsymmetric trees.
        # cs.add_condition(
        #     EqualsCondition(cs["boosting_type"], cs["grow_policy"], "SymmetricTree")
        # )

        # PerTreeLevel sampling is not supported for Lossguide grow policy.
        # cs.add_condition(
        #     NotEqualsCondition(cs["sampling_frequency"], cs["grow_policy"], "Lossguide")
        # )

        return cs

    elif algo == GradientBoostingClassifier:
        return ConfigurationSpace(
            {
                "ccp_alpha": (0.0, 1.0),
                "learning_rate": (0.001, 1.0),
                "loss": ["log_loss", "exponential"],
                "max_depth": (1, 50),
                "max_features": ["sqrt", "log2", "None"],
                "max_leaf_nodes": (2, 50),
                "min_impurity_decrease": (0.0, 2.0),
                "min_samples_leaf": (0.0001, 0.9999),
                "min_samples_split": (0.0001, 1.0),
                "min_weight_fraction_leaf": (0.0, 0.5),
                "n_estimators": (1, 300),
                # "n_iter_no_change": None, -- No early stopping for now
                # "random_state": None, -- Set by SMAC
                "subsample": (0.0001, 1.0),
                # "tol": 0.0001, -- No early stopping for now
                # "validation_fraction": 0.1, -- No early stopping for now
            }
        )

    elif algo == MLPClassifier:
        return ConfigurationSpace(
            {
                "activation": ["logistic", "tanh", "relu"],
                "alpha": (0.00001, 0.001),
                # "batch_size": "auto",
                # "beta_1": 0.9,
                # "beta_2": 0.999,
                # "early_stopping": False,
                # "epsilon": 1e-08,
                "hidden_layer_sizes": (50, 200),
                "learning_rate": ["constant", "invscaling", "adaptive"],
                "learning_rate_init": (0.001, 0.1),
                # "max_fun": 15000,
                # "max_iter": 200,
                "momentum": (0.0, 1.0),
                # "n_iter_no_change": 10,
                # "nesterovs_momentum": True,
                # "power_t": 0.5,
                # "random_state": None, -- Set by SMAC
                # "shuffle": True,
                "solver": ["lbfgs", "sgd", "adam"],
                # "tol": 0.0001,
                # "validation_fraction": 0.1,
                # "verbose": False,
                # "warm_start": True, -- constant bool not allowed, set this manually
            }
        )

    elif algo == LGBMClassifier:
        return ConfigurationSpace(
            {
                "boosting_type": ["gbdt", "rf", "dart"],
                # "class_weight": None, -- SMOTE instead
                "colsample_bytree": (0.0, 1.0),
                "learning_rate": (0.01, 0.5),
                "max_depth": (-1, 10),
                "min_child_samples": (10, 40),
                "min_child_weight": (0.001, 0.1),
                "min_split_gain": (0.0, 0.5),
                "n_estimators": (50, 300),
                "num_leaves": (15, 128),
                # "random_state": None, -- set by SMAC
                "reg_alpha": (0.0001, 0.1),
                "reg_lambda": (0.0001, 0.1),
                "subsample": (0.0, 1.0),
                "subsample_for_bin": (100000, 300000),
                "subsample_freq": (0, 200),
            }
        )

    else:
        raise ValueError("Unknown algorithm!")
