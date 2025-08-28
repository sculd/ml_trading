import logging
import sys
import os, datetime, pprint, itertools
import setup_env # needed for the environment variables

import pandas as pd
import numpy as np
import mlflow
from mlflow import MlflowClient


_tracking_uri = "http://100.108.193.31:8080"
mlflow.set_tracking_uri(_tracking_uri)

if __name__ == '__main__':
    experiments = mlflow.search_experiments()
    print([e.name for e in experiments])

    client = MlflowClient(tracking_uri=_tracking_uri)
    crypto_experiment = client.get_experiment_by_name("crypto_backtest")

    runs = []
    page_token = None
    while True:
        paged_runs = client.search_runs([crypto_experiment.experiment_id], page_token=page_token)
        runs += paged_runs.to_list()
        if paged_runs.token is None:
            break
        page_token = paged_runs.token

    for run in runs:
        print(f"{run.info.run_id=}\n{run.info.status=}\n{run.data.metrics=}\n")
        #client.delete_run(run.info.run_id)

    print(f"total runs: {len(runs)}")
    # example run:
    # http://100.108.193.31:8080/#/experiments/737325085219991395/runs/1f8c89ead70940fd9e93b530bbdca0f3

