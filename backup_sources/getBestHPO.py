import optuna

storage_url='postgresql://jlopez:dbpass@localhost/optuna_study'
study_name = 'myresearch_study_optimizer'
study = optuna.load_study(study_name=study_name, storage=storage_url)

print("HPO Study:", study_name)
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

