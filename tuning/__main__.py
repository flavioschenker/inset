import random
import numpy
import torch
import wandb
from benchmark.parameters import parameters
from benchmark.hyperparameters import hyperparameters
from data.dataset import Dataset
from tuning.experiment import Experiment

param = parameters()
debug = param["debug"]
sweep = param["sweep"]
task = param["task"]
wentity = param["wandb_entity"]
jobid = param["jobid"]
seed = param["seed"]

random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)

if debug:
    print("debug mode")
    if param["wandbosity"]:
        wandb.init(project="tuning_"+str(task), entity=wentity, config=param, name=str(jobid))
    dataset = Dataset(param, debug=True)
    testset = Dataset(param, debug=True, validation=True)
    experiment = Experiment(param, dataset, testset)
    experiment.run()
elif sweep:
    print("sweep mode")
    param["wandbosity"] = 2
    dataset = Dataset(param, sweep=True)
    testset = Dataset(param, sweep=True, validation=True)
    def run_sweep():
        wandb.init(allow_val_change=True)
        config = wandb.config
        experiment = Experiment(config, dataset, testset)
        experiment.run()
    sweep_id = wandb.sweep(hyperparameters(param), project="sweep_"+str(task), entity=wentity)
    wandb.agent(sweep_id=sweep_id, function=run_sweep)
else:
    if param["wandbosity"]:
        wandb.init(project="tuning_"+str(task), entity=wentity, config=param, name=str(jobid))
    dataset = Dataset(param)
    testset = Dataset(param, validation=True)
    experiment = Experiment(param, dataset, testset)
    experiment.run()