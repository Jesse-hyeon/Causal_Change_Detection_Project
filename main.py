import sys
import os
sys.path.append(os.path.abspath("src"))  # src 폴더를 경로에 추가

import json
import random
import numpy as np

import torch

from train.exp import Exp_Main

# JSON 파일 경로
config_path = "/Users/choeseoheon/Desktop/Causal-Discovery/src/Arg/config.json"

# JSON 파일 로드
with open(config_path, "r") as f:
    config = json.load(f)

fix_seed = config["seed"]
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

len_in = len(config["feature_set"]) - 1
# config (json file 안에 있는 enc_in, dec_in len_in으로 변경하는 코드
config["enc_in"] = len_in
config["dec_in"] = len_in

def run(config):
    exp = Exp_Main(config)
    exp.train(setting='custom_experiment')
    exp.test(setting='custom_experiment')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run(config)
