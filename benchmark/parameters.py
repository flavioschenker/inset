import argparse
initial_parameters = {
    "validation_split": 0.01,
    "sweep_fraction": 0.01,
    "dim_sequence": 50,
    "dim_classes": 7,
    "seed": 31122022,
}
debug_parameters = {
    "debug_dataset_fraction": 0.001,
    "debug_training_fraction": 0.06,
    "debug_validation_fraction": 0.06,
    "debug_epochs": 1,
    "debug_train_batch_size": 16,
    "debug_valid_batch_size": 16,
}
default_parameters = {
    # general
    "directory": "/itet-stor/flaviosc/net_scratch/sp_data/",
    "wandb_entity": "berseqr",
    "train_model": True,
    "eval_model": True,
    "save_model": True,
    "train_batch_size": 120,
    "valid_batch_size": 120,
    "optimizer": "adam",
    # transformer
    "transformer_normalising": True,
    "transformer_position_encoding": True,
    # task unmasking
    "unmasking_mask_probability": 0.25,
    "unmasking_regressive_dim_layers": [2000,4000,2000],
    "unmasking_contrastive_margin": 10,
    "unmasking_contrastive_dim_layers": [2000,4000],
    "unmasking_contrastive_dim_embedding_space": 8000,
    # task continuation
    "continuation_mask_length": 5,
    "continuation_contrastive_margin": 1,
    "continuation_neg_pairs_fraction": 2,
    "continuation_contrastive_dim_layers": [200,100],
    "continuation_dim_embedding_space": 50,
    "continuation_regressive_dim_layers": [2000,4000,2000],
    # task similarity
    "similarity_contrastive_margin": 1,
    "similarity_pos_pairs_batch_multiply": 2,
    "similarity_neg_paris_batch_multiply": 2,
    "similarity_dim_layers": [100],
    "similarity_dim_embedding_space": 20,
    # task classification
    "classification_confidence_threshold": 0.55,
    "classification_dim_layers": [200,100],
    # task complexity
    "complexity_dim_layers": [200,100],
    # task next
    "next_dim_layers": [200,100,20],
}
def parameters():
    parser = setup_parser()
    return {**initial_parameters, **debug_parameters, **default_parameters, **vars(parser.parse_args())}
def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=12,
        help="number of epochs."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate (default=1e-4)"
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.99,
        help="exponential learning rate decay (default=0.99)"
    )
    parser.add_argument(
        "--l2_regularization",
        type=float,
        default=0.0,
        help="weight decay regularization (default=1e-5)"
    )
    parser.add_argument(
        "--training_fraction",
        type=float,
        default=1.0,
        help="fraction to use of the training dataset."
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.1,
        help="fraction to use of the validation dataset."
    )
    parser.add_argument(
        "--leave_one_out",
        type=str,
        choices=["unmasking", "classification","complexity","next"],
        help="pre-training-task to leave out for leave-one-out-analysis."
    )
    parser.add_argument(
        "--leave_one_out_feature",
        type=str,
        choices=["position", "sign","normalized","log","digits","mask"],
        help="pre-training-features to leave out for leave-one-out-analysis."
    )
    parser.add_argument(
        "--transformer_dim_heads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--transformer_dim_layers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--transformer_dim_feedforward",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--transformer_dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
		"-d",
		"--debug",
		action="store_true",
		help="debug mode with reduced dataset"
	)
    parser.add_argument(
		"-s",
		"--sweep",
		action="store_true",
		help="sweep mode with hyperparameter tuning"
	)
    parser.add_argument(
        "-r",
        "--regime",
        type=str,
        default="attask",
        choices=["inbatch", "inepoch","atepoch","attask"],
        help="choose the training-regime (default=attask)"
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=["unmasking", "classification","complexity","next","similarity","continuation"],
        help="choose which task to use."
    )
    parser.add_argument(
		"-c",
		"--contrastive",
		action="store_true",
		help="contrastive version of the task"
    )
    parser.add_argument(
		"-j",
		"--jobid",
		type=str,
		default="local",
		help="the job id (and the id of the wandb run)"
	)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="new",
        help="model id or new model (default=new)"
    )
    parser.add_argument(
        "-f",
        "--model_freeze",
    	action="store_true",
		help="freeze parameters of a prepended model"
)
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=1,
        choices=[0,1,2,3],
        help="level of verbosity for the console (0 none, 1 min, 2 max, 3 model, default 0)"
    )
    parser.add_argument(
        "-w",
        "--wandbosity",
        type=int,
        default=0,
        choices=[0,1,2],
        help="level of verbosity for wandb (0 none, 1 min, 2 max, default 0)"
    )
    return parser