import logging
import os
import re
import sys
from datetime import datetime as time

import torch
import wandb

from utils import PROJECT_NAME, PROJECT_PATH, ENTITY_NAME


def validate_wandb_name(project_name: str, name: str) -> str:
    api = wandb.Api()
    runs = api.runs(project_name)
    original_name = name

    def existing(n):
        for run in runs:
            if run.name == n:
                return True
        return False

    i = 1
    while existing(name):
        i += 1
        name = f"{original_name}_{i}"

    return name


class Logger:
    def __init__(self, wb_name=None, verbose=False, group=None, conf=None, device=None, use_wandb=True):
        resume = conf['setup_args']['resume']
        self._use_wandb = use_wandb
        if resume:
            self.run_id = conf['setup_args']['run_id']
            if self.run_id is None:
                raise ValueError(
                    "If 'resume' is True, 'run_id' must be non-null")
        else:
            self.run_id = wandb.util.generate_id()
            conf['setup_args']['run_id'] = self.run_id

        self.logger = logging.getLogger(PROJECT_NAME)
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

        # timestamp = time.now().strftime("%Y-%m-%d_%H-%M-%S")
        # foldname = timestamp[:10]  # get year, month and day
        # logdir = os.path.join(PROJECT_PATH, "logs", foldname)
        # os.makedirs(logdir, exist_ok=True)
        # log_path = os.path.join(logdir, f"{timestamp}.log")

        logdir = os.path.join(PROJECT_PATH, "logs")
        os.makedirs(logdir, exist_ok=True)
        # TODO: check if file already exists (?)
        self.log_path = os.path.join(logdir, f"{self.run_id}.log")
        if resume and not os.path.isfile(self.log_path):
            self._restore_local_log()
            # after this call, there should be in the log folder the `log_path` file

        file_format = "[%(asctime)s] - [%(levelname)s]: %(message)s"
        stdout_format = "[%(levelname)s]: %(message)s"

        # TODO: looks like the file handler natively prints on the previous file if it exists
        self.file_handler = logging.FileHandler(self.log_path)
        self.file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(file_format)
        self.file_handler.setFormatter(file_formatter)

        self.stdout_handler = logging.StreamHandler(stream=sys.stdout)
        self.stdout_handler.setLevel(logging.INFO)
        stdout_formatter = logging.Formatter(stdout_format)
        self.stdout_handler.setFormatter(stdout_formatter)

        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.logger.setLevel(logging.INFO)

        self.verbose = verbose

        if self._use_wandb:
            if resume:
                self.wandb_name = wb_name
                if self.wandb_name is None:
                    raise ValueError(
                        "Run name cannot be null if resume is True.")
            else:
                w_name = wb_name if wb_name is not None else f'run_{time.now().strftime("%Y-%m-%d_%H-%M-%S")}'

                self.wandb_name = validate_wandb_name(
                    project_name=conf['name'], name=w_name)

                if self.wandb_name != w_name:
                    self.logger.warning(
                        f"Run {w_name} already exists. Using name {self.wandb_name} instead.")

            wandb_settings = wandb.Settings(
                silent="true") if not self.verbose else None

            wandb_path = os.path.join(PROJECT_PATH, '.wandb')
            os.makedirs(wandb_path, exist_ok=True)
            wandb.init(
                entity=ENTITY_NAME,
                project=conf['name'],
                name=self.wandb_name,
                id=self.run_id,
                dir=wandb_path,
                settings=wandb_settings,
                group=group,
                config=conf,
                resume="allow"
            )

            # Before logging anything, print the previous log file, if any
            if resume:
                self._restore_wandb_log()

            self.log(f"Using run_id {self.run_id}")

            self.ckpt_path = os.path.join(
                PROJECT_PATH, "checkpoints", self.wandb_name)
            os.makedirs(self.ckpt_path, exist_ok=True)

    def log(self, message, verbose=None, level: int = logging.INFO):
        verbose = verbose if verbose is not None else self.verbose
        if not verbose:
            self.logger.removeHandler(self.stdout_handler)
        if type(message) is str:
            self.logger.log(level, message)
        elif type(message) is dict:
            string_message = f"Episode {message['episode']} - " if 'episode' in message else ''
            sm = []
            for key, value in message.items():
                key = key.split('/')[-1]
                if key == 'episode':
                    continue
                elif isinstance(value, float):
                    sm.append(f'{key.capitalize()}: {value:.5f}')
                elif isinstance(value, int):
                    sm.append(f'{key.capitalize()}: {int(value)}')

            string_message += ', '.join(sm)
            self.logger.log(level, string_message)
            if self._use_wandb:
                wandb.log(message)
        else:
            raise TypeError(f"Cannot log an object of type {type(message)}")
        if not verbose:
            self.logger.addHandler(self.stdout_handler)

    def save_checkpoint(self, data, fname="ckpt.pt"):
        # Locally the checkpoints are overwritten while remotely they are versioned
        # TODO: Every 5 overwrites, save a non overwritable one (?)

        if self._use_wandb:
            save_path = os.path.join(self.ckpt_path, fname)
            torch.save(data, save_path)
            wandb.save(save_path)

    def load_checkpoint(self, fname="ckpt.pt"):
        if self._use_wandb:
            data_path = wandb.restore(
                fname, root=self.ckpt_path, replace=False)
            data_path = data_path.name
        else:
            data_path = fname
        try:
            data = torch.load(data_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file {data_path} not found.")
        return data

    def _restore_local_log(self):
        log_path = os.path.join(PROJECT_PATH, "logs")
        data_path = wandb.restore("output.log",
                                  run_path=f"{ENTITY_NAME}/{PROJECT_NAME}/{self.run_id}",
                                  root=log_path,
                                  replace=False)
        os.rename(data_path.name, os.path.join(log_path, f"{self.run_id}.log"))

    def _restore_wandb_log(self):
        pattern = r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\] - '
        with open(self.log_path) as previous_log:
            for line in previous_log.readlines():
                # Use re.sub to remove the timestamp and hyphen if the pattern is found
                print(re.sub(pattern, '', line)
                      if re.match(pattern, line) else line)


if __name__ == '__main__':
    logger = Logger()
    logger.log("Ciao", verbose=False)
