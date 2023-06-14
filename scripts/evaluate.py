import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


import numpy as np
from collections import deque, namedtuple
import random
import math
from itertools import count

import os
import pickle
from datetime import datetime

import glob
import argparse

import q_learning as ql

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=False, default='0')

    opt = parser.parse_args()

    model = int(opt.model)


    if model == 0:
        filepath = './models/TactileOnly'
        reduced = 3
        pitch_only = False
        sensor = 'fingertip'
    if model == 1:
        filepath = './models/Tactile+Torques+EEorientation'
        reduced = 1
        pitch_only = False
        sensor = 'fingertip'
    if model == 2:
        filepath = './models/Fullstate'
        reduced = 0
        pitch_only = False
        sensor = 'fingertip'
    if model == 3:
        filepath = './models/BigSensor'
        reduced = 2
        pitch_only = False
        sensor = 'plate'


    algo = ql.DQN_Algo(filepath=filepath, reduced=reduced, sensor=sensor, pitch_only=pitch_only, eval=True)
    algo.max_ep_steps = 150
    algo.evaluate()
