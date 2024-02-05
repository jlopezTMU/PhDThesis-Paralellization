## This program simply deletes an Optuna study

import optuna

# Assuming you're using a PostgreSQL database for storage
storage_url='postgresql://jlopez:dbpass@localhost/optuna_study'
study_name = 'myresearch_study_optimizer'

# Deleting the study
print("Deleting the Optuna study: ", study_name)
optuna.delete_study(study_name=study_name, storage=storage_url)
