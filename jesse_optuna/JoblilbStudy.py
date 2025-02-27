import copy
import optuna
import numpy as np
from dask.distributed import Client , as_completed
import yaml
import pathlib

class JoblibStudy:
    def __init__(self, **study_parameters):
        self.get_config()
        self.study_parameters = study_parameters
        self.study: optuna.study.Study = optuna.create_study(**study_parameters)

    def _optimize_study(self, func, n_trials, **optimize_parameters):
        study_parameters = copy.copy(self.study_parameters)
        study_parameters["study_name"] = self.study.study_name
        study_parameters["load_if_exists"] = True
        study = optuna.create_study(**study_parameters)
        study.sampler.reseed_rng()
        study.optimize(func, n_trials=n_trials, **optimize_parameters, catch=(Exception,))

    def optimize(self, func, n_trials=1, **optimize_parameters):
        # Create a Dask client
        client = Client(f'tcp://{self.dask_ip}:{self.dask_port}')
        print("Connected to Dask client.")

        # Submit all tasks at once and let Dask handle the distribution
        futures = [client.submit(self._optimize_study, func,pure=False, n_trials=1, **optimize_parameters)
                   for _ in range(n_trials)]

        # Wait for the tasks to complete and gather the results
        for future in as_completed(futures):
            result = future.result()  # Handle results here if needed
            print(f"Result: {result}")

        client.close()

    def set_user_attr(self, key: str, value):
        if isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        self.study.set_user_attr(key, value)

    def __getattr__(self, name):
        if not name.startswith("_") and hasattr(self.study, name):
            return getattr(self.study, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    def get_config(self):
        cfg_file = pathlib.Path('optuna_config.yml')

        if not cfg_file.is_file():
            print("optuna_config.yml not found. Run create-config command.")
            exit()
        else:
            with open("optuna_config.yml", "r") as ymlfile:
                cfg = yaml.load(ymlfile, yaml.SafeLoader)
                
        self.dask_ip = cfg['dask_scheduler_ip']
        self.dask_port = cfg['dask_scheduler_port']

        return cfg
