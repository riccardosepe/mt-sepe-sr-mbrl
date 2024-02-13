import random

import numpy as np
import torch


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


# function to convert a value of seconds to a string of format "minutes: mm, seconds: ss.milliseconds"
def sec_to_min(seconds):
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}min, {seconds:.2f}s"
