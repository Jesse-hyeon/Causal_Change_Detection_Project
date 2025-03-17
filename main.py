import sys
import os
sys.path.append(os.path.abspath("src"))  # src 폴더를 경로에 추가

import json
import random
import numpy as np

import torch

from train.exp_former import Exp_Main_former
from train.exp_rnn import Exp_Main_rnn

# JSON 파일 경로
config_path = "/Users/choeseoheon/Desktop/Causal-Discovery/src/Arg/config.json"

# JSON 파일 로드
with open(config_path, "r") as f:
    config = json.load(f)

fix_seed = config["seed"]
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

len_in_former = len(config["feature_set"]) - 1
len_in_rnn= len(config["feature_set"]) - 1 + 3

def run(config):
    if config["model"] in ['transformer', 'ns_Transformer']:
        config["enc_in"] = len_in_former
        config["dec_in"] = len_in_former
        exp = Exp_Main_former(config)
        exp.train(setting='custom_experiment')
        exp.test(setting='custom_experiment')
    else:
        config["enc_in"] = len_in_rnn
        config["dec_in"] = len_in_rnn
        exp = Exp_Main_rnn(config)
        exp.train(setting='custom_experiment')
        exp.test(setting='custom_experiment')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run(config)
