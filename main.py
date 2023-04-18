from models.model_ctgan import CTGANSynthesizer
from models.model_dpctgan import DPCTGANSynthesizer

import ctgan
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from utils.utils import eval_dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from utils.privacy import normalized_avg_dist, avg_dist
from utils.fidelity import get_predictions, eval_fidelity

import torch

class Main:
    def __init__(self, dataset, target):
        print("Initializing class...")
        self.dataset = dataset
        self.target = target
        self.ctgan = CTGANSynthesizer(epochs=5)
        self.dpctgan = DPCTGANSynthesizer(verbose=True,
                            epochs=1,
                            clip_coeff=0.1,
                            sigma=4,
                            target_epsilon=2, #4
                            target_delta=1e-5
                            )

    def prepare_data(self, sep=',', drop_columns=None):
        print("Preparing data...")

        df = pd.read_csv(self.dataset, sep = sep)

        if drop_columns is not None:
            df = df.drop(columns=[drop_columns])

        df = df.apply(pd.to_numeric)
        
        df.loc[df["y"] > 1 , "y"] = 0   #TODO

        df = (df-df.min())/(df.max()-df.min())

        X = df.drop(columns=[self.target])
        y = df[self.target]

        return df, X, y

    def train(self, data):
        print("Training models...")
        self.ctgan.fit(data)
        self.dpctgan.fit(data)

    def save_models(self, dataset_name):
        torch.save(self.ctgan, './saved_models/ctgan' + dataset_name + '.pt')
        torch.save(self.dpctgan, './saved_models/dpctgan' + dataset_name + '.pt')

    def load_models(self, dataset_name):
        self.ctgan = torch.load('./saved_models/ctgan' + dataset_name + '.pt')
        self.dpctgan = torch.load('./saved_models/dpctgan' + dataset_name + '.pt')


    def generate(self, model):
        if model == 'ctgan':
            samples = self.ctgan.sample(len(self.dataset))
        elif model == 'dpctgan':
            samples = self.dpctgan.sample(len(self.dataset))
        else:
            print('Error: model not found')

        X_syn = samples.drop(columns=[self.target])
        y_syn = samples[self.target]
        y_syn = y_syn.round(0)
        y_syn = y_syn.astype(int)

        return samples, X_syn, y_syn
    
    def calculate_privacy(self, synthetic_data, real_data):
        synthetic_data = synthetic_data.apply(pd.to_numeric) # convert all columns of DataFrame
        real_data = real_data.apply(pd.to_numeric) # convert all columns of DataFrame

        return avg_dist(synthetic_data, real_data)
    
    def calculate_fidelity(self, X_real_data, y_real_data, X_synthetic_data, y_synthetic_data):
        X_train, X_test, y_train, y_test = train_test_split(X_real_data, y_real_data, test_size=0.3, random_state=42)

        X_syn_train, X_syn_test, y_syn_train, y_syn_test = train_test_split(X_synthetic_data, y_synthetic_data, test_size=0.3, random_state=42)

        rr_pred = get_predictions(X_train, y_train, X_test, y_test)
        fr_pred = get_predictions(X_syn_train, y_syn_train, X_test, y_test)
        #rf_pred = get_predictions(X_train, y_train, X_syn_test, y_syn_test)
        #ff_pred = get_predictions(X_syn_train, y_syn_train, X_syn_test, y_syn_test)

        return eval_fidelity(rr_pred, fr_pred)
    







