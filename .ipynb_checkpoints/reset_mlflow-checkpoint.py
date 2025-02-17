import mlflow

# Forcefully end all active MLFlow runs
while mlflow.active_run():
    print("Ending an active MLFlow run...")
    mlflow.end_run()

print(" All active MLFlow runs have been closed.")