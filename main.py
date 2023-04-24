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
                            target_epsilon=1, #4
                            target_delta=1e-5
                            )

    def prepare_data(self, sep=',', drop_columns=None, flag=0):
        print("Preparing data...")

        if flag == 2:
            df = pd.read_csv(self.dataset, sep =";")
        else:
            df = pd.read_csv(self.dataset, sep = sep)

        if drop_columns is not None:
            df = df.drop(columns=[drop_columns])

        df = df.apply(pd.to_numeric)
        
        if flag == 1:
            df.loc[df["y"] > 1 , "y"] = 0

        df = (df-df.min())/(df.max()-df.min())

        X = df.drop(columns=[self.target])
        y = df[self.target]

        return df, X, y

    def train(self, data):
        print("Training models...")
        self.ctgan.fit(data)
        self.dpctgan.fit(data)

    def save_models(self, dataset_name):
        if not os.path.exists('./saved_models'):
            os.makedirs('./saved_models')

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
    
    def save_results(self, name, fidelity, privacy):
        if not os.path.exists('./results'):
            os.makedirs('./results')
        
        # combine the fidelity and privacy lists into a list of tuples
        results = list(zip(fidelity, privacy))

        # open the file in write mode and write the results to it
        with open(f"./results/{name}.txt", "w") as f:
            for fidelity_val, privacy_val in results:
                f.write(f"{fidelity_val}\t{privacy_val}\n")

import sys

if __name__ == '__main__':
    dataset = sys.argv[1]
    target = sys.argv[2]
    drop_columns = sys.argv[3]
    flag = int(sys.argv[4])
    name = sys.argv[5]

    run = Main(dataset, target)

    if drop_columns == 'None':
        drop_columns = None

    data, X, y = run.prepare_data(drop_columns=drop_columns, flag=flag)

    run.train(data)

    run.save_models(name)

    ctgan_samples, ctgan_X_syn, ctgan_y_syn  = run.generate('ctgan')
    dpctgan_samples, dpctgan_X_syn, dpctgan_y_syn = run.generate('dpctgan')

    ctgan_privacy = run.calculate_privacy(ctgan_samples, data)
    dpctgan_privacy = run.calculate_privacy(dpctgan_samples, data)

    ctgan_fidelity, _, _ = run.calculate_fidelity(X, y, ctgan_X_syn, ctgan_y_syn)
    dpctgan_fidelity, _, _ = run.calculate_fidelity(X, y, dpctgan_X_syn, dpctgan_y_syn)


    fidelity = [ctgan_fidelity, dpctgan_fidelity]
    privacy = [ctgan_privacy, dpctgan_privacy]

    run.save_results(name, fidelity, privacy)


    







