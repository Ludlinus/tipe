import wandb
import run_NN

wandb.login(key="1b1a372f1e56f15f8c0a2fd3cd93c5f5f5ca1dae")

sweep_configuration = {
    "method": "bayes",
    "name": "sweep_3",
    "metric": {"goal": "maximize", "name": "fit_mean"},
    "parameters": {
        "node_add_prob": {"max": 0.01, "min": 0.001},
        "node_delete_prob": {"max": 0.005, "min": 0.0005},
        "conn_add_prob": {"max": 0.5, "min": 0.05},
        "conn_delete_prob": {"max": 0.25, "min": 0.025},
        "bias_max_value": {"max": 40, "min": 20},
        "weight_max_value": {"max": 100, "min": 10},
        "survival_threshold": {"max": 0.30, "min": 0.10}
    },
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project="TIPE-2", entity="sweat_pas_rose")

run_NN.main()
