import wandb

wandb.login(key="1b1a372f1e56f15f8c0a2fd3cd93c5f5f5ca1dae")

sweep_configuration = {
    "method": "bayes",
    "name": "sweep_5",
    "metric": {"goal": "maximize", "name": "fit_mean"},
    "parameters": {
        "node_add_prob": {"max": 0.1, "min": 0.001},
        "node_delete_prob": {"max": 0.05, "min": 0.001},
        "survival_threshold": {"max": 0.40, "min": 0.05},
        "pop_size": {"min": 450, "max": 550}
    },
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="TIPE-2", entity="sweat_pas_rose")
