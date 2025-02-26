hyperparameters_Transformer = {
    "name": "Transformer",
    "method": "grid",
    "metric": {
        "name": "val_mae_of_logs",
        "goal": "minimize",
    },
    "parameters": {
        # "transformer_encoder_layers": {
        #     "values": [4,8],
        # },
        # "transformer_attention_heads": {
        #     "values": [4],
        # },
        # "transformer_feedforward_dim": {
        #     "values": [2000],
        # },
        # "transformer_dropout": {
        #     "values": [0.001, 0.0]
        # },
        "lr": {
            "values": [1e-3,1e-4]
        },
        "lr_decay": {
            "values": [0.5,0.7,0.9]
        },
        # "transformer_positional_encoding": {
        #     "values": [True, False]
        # }
    }
}

def get_hyperparameters(parameters, hyperparameters):
    p = {}
    result = hyperparameters.copy()
    result["name"] = result["name"] + "_" + str(parameters["jobid"])
    # reshape to wandb-format
    for key, value in parameters.items():
        p[key] = {
            "value": value
        }
    result["parameters"] = p # append parameters to result
    # override hyperparameters
    for key, value in hyperparameters["parameters"].items():
        result["parameters"][key] = value
    return result


def hyperparameters(parameters):
    return get_hyperparameters(parameters, hyperparameters_Transformer)