# File: src/models/ann.py
import pandas as pd
import numpy as np
import csv
import os
import streamlit as st

import torch
import itertools
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from efficient_kan import KAN
import torch.nn as nn

from sklearn.model_selection import KFold, ParameterGrid
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from src.models.base_model import BaseMLModel
from src.utils.scaling_utils import mape, r_squared, ytest_to_initial_scale

class KANModel(BaseMLModel):
    """Kolmogrov-Arnold Network model implementation."""

    def __init__(self):
        self.model = None
        self.best_params = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    @property
    def name(self) -> str:
        return "Kolmogrov-Arnold Network"

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """Train the ANN model with the best parameters."""
        params = kwargs.get("params", self.best_params if self.best_params else {})
        # default_layers = [[X_train.shape[1]], [X_train.shape[1], X_train.shape[1]]]

        bestparameter_csv_file = 'outputs/KAN_bestparams.csv'
        folder_path = os.path.dirname(bestparameter_csv_file)
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if os.path.isfile(bestparameter_csv_file):
            os.remove(bestparameter_csv_file)
    
        def write_csv(results_dict, csv_file):
            headers = list(results_dict.keys())
            rows = zip(*results_dict.values())

            with open(csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                # Write the header row (dictionary keys)
                writer.writerow(headers)
                # Write each row of values
                for row in rows:
                    writer.writerow(row)
        
        param_names = list(params.keys())
        results_dict = {key: [] for key in param_names}
        for name in param_names:
            results_dict[name].append(params[name])
        write_csv(results_dict, bestparameter_csv_file)
        
        X_train_array = X_train.to_numpy()
        y_train_array = y_train.to_numpy()
        
        x_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_array, dtype=torch.float32)

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        
        trainloader = DataLoader(train_dataset, batch_size=params.get('batch_size'), shuffle= False)    
        input_size = X_train_array.shape[1]
    
        if len(params.get('hidden_layer_size'))==1:
            # print([input_size, sub[0], 1])
            self.model = KAN([input_size, params.get('hidden_layer_size')[0], 1], grid_size=params.get('grid_size'), spline_order=params.get('spline_order'))
            # model = KAN([input_size, hidden_layer_size[0], 1], spline_order=spline_order)
        else:
            # print([input_size, *sub, 1])
            self.model = KAN([input_size, *params.get('hidden_layer_size'), 1], grid_size=params.get('grid_size'), spline_order=params.get('spline_order'))
            # model = KAN([input_size, *hidden_layer_size, 1], spline_order=spline_order)
        
        self.model.to(self.device)
        weight_decay = 0
        optimizer = optim.AdamW(self.model.parameters(), lr=params.get('lr'), weight_decay=weight_decay)
        # optimizer = optim.AdamW(self.model.parameters(), lr=params.get('lr'), weight_decay=params.get('weight_decay'))
        points_param = 0
        
        # train model
        # print("## Training")
        self.model.train()
        for epoch in range(params.get('epochs')):
            for batch_x_train, batch_y_train in trainloader:
                batch_x_train, batch_y_train = batch_x_train.to(self.device), batch_y_train.to(self.device)
                optimizer.zero_grad()
                predictions_train = self.model(batch_x_train)
                # loss = nn.functional.mse_loss(predictions_train, batch_y_train) + params.get('points') * self.model.regularization_loss()
                loss = nn.functional.mse_loss(predictions_train.squeeze(1), batch_y_train) + points_param * self.model.regularization_loss()
                loss.backward()
                optimizer.step()    


    def predict(self, X: pd.DataFrame, **kwargs) -> pd.Series:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        else:
            
            self.model.eval()
            with torch.no_grad():
                predictions_test, truth_test = [], []
                
                X_test_array = X.to_numpy()
                x_test_tensor = torch.tensor(X_test_array, dtype=torch.float32)
        
                preds_test = self.model(x_test_tensor.to(self.device))
            
        return pd.Series(preds_test.flatten(), index=X.index)


    def tune(self, X_train: pd.DataFrame, y_train: pd.Series,
             param_grid: dict, k_folds: int, scalers: dict, **kwargs) -> pd.DataFrame:
        """Tune hyperparameters using grid search and cross-validation."""

        # Add num_of_layers_neurons to param_grid
        # layers_configs = [[X_train.shape[1]], [X_train.shape[1], X_train.shape[1]]]
        # if 'hidden_layer_size' not in param_grid:
        #     param_grid['hidden_layer_size'] = layers_configs
        # else:
        #     # Ensure layers_configs are included
        #     param_grid['hidden_layer_size'] = [
        #         config for config in param_grid['hidden_layer_size'] if config in layers_configs
        #     ] or layers_configs

        hyperparameter_csv_file = 'outputs/KAN_hyperparametertuning.csv'
        folder_path = os.path.dirname(hyperparameter_csv_file)
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if os.path.isfile(hyperparameter_csv_file):
            os.remove(hyperparameter_csv_file)
    
        def write_csv(results_dict, csv_file):
            headers = list(results_dict.keys())
            rows = zip(*results_dict.values())

            with open(csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                # Write the header row (dictionary keys)
                writer.writerow(headers)
                # Write each row of values
                for row in rows:
                    writer.writerow(row)

        param_names = list(param_grid.keys())
        
        param_grid['hidden_layer_size'] = [
            [int(x) for x in item.strip('[]').split()]
            for item in param_grid['hidden_layer_size']
        ]
        
        
        results_dict = {key: [] for key in param_names}
        results_dict.update({"Val_MAPE": [], "Val_R2": []})
        results_dict.update({"Input_features": [], "Num_of_parameters": []})

        param_combinations = list(ParameterGrid(param_grid))
        
        # st.write('param_grid', param_grid.get('hidden_layer_size'))
        # st.write('param_combinations', param_combinations)
                
        total_combinations = len(param_combinations)

        st.write(f"Total parameter combinations: **{total_combinations}**")
        status = st.status("Starting tuning process...", expanded=False)
        progress_bar = st.progress(0)

        # for params in param_combinations:
        for i, params in enumerate(param_combinations):
            status.update(label=f"Testing Combination {i+1}/{total_combinations}")
            kf = KFold(n_splits=k_folds)
            val_mape, val_r2 = [], []
            # st.write('start grid', params.get('hidden_layer_size'))
            for train_idx, val_idx in kf.split(X_train):
                # st.write('fold')
                X_train_array = X_train.iloc[train_idx].to_numpy()
                y_train_array = y_train.iloc[train_idx].to_numpy()
                
                x_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train_array, dtype=torch.float32)

                X_test_array = X_train.iloc[val_idx].to_numpy()
                y_test_array = y_train.iloc[val_idx].to_numpy()
                
                x_test_tensor = torch.tensor(X_test_array, dtype=torch.float32)
                y_test_tensor = torch.tensor(y_test_array, dtype=torch.float32)
                
                train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
                test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
                
                trainloader = DataLoader(train_dataset, batch_size=params.get('batch_size'), shuffle= False)   
                testloader = DataLoader(test_dataset, batch_size=params.get('batch_size'), shuffle= False) 
                
                input_size = X_train_array.shape[1]
                
                if len(params.get('hidden_layer_size'))==1:
                    # print([input_size, sub[0], 1])
                    self.model = KAN([input_size, params.get('hidden_layer_size')[0], 1], grid_size=params.get('grid_size'), spline_order=params.get('spline_order'))
                    # model = KAN([input_size, hidden_layer_size[0], 1], spline_order=spline_order)
                else:
                    # print([input_size, *sub, 1])
                    self.model = KAN([input_size, *params.get('hidden_layer_size'), 1], grid_size=params.get('grid_size'), spline_order=params.get('spline_order'))
                    # model = KAN([input_size, *hidden_layer_size, 1], spline_order=spline_order)
                
                self.model.to(self.device)
                
                num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                         
                weight_decay = 0
                optimizer = optim.AdamW(self.model.parameters(), lr=params.get('lr'), weight_decay=weight_decay)
                # optimizer = optim.AdamW(self.model.parameters(), lr=params.get('lr'), weight_decay=params.get('weight_decay'))
                points_param = 0
                
                # train model
                # print("## Training")
                self.model.train()
                for epoch in range(params.get('epochs')):
                    for batch_x_train, batch_y_train in trainloader:
                        batch_x_train, batch_y_train = batch_x_train.to(self.device), batch_y_train.to(self.device)
                        optimizer.zero_grad()
                        predictions_train = self.model(batch_x_train)
                        loss = nn.functional.mse_loss(predictions_train.squeeze(1), batch_y_train) + points_param * self.model.regularization_loss()
                        # loss = nn.functional.mse_loss(predictions_train.squeeze(1), batch_y_train) + params.get('points') * self.model.regularization_loss()
                        loss.backward()
                        optimizer.step()  
                
                # test
                # print("## Testing")
                self.model.eval()
                with torch.no_grad():
                    predictions_test, truth_test = [], []
                    for batch_x_test, batch_y_test in testloader:
                        batch_x_test, batch_y_test = batch_x_test.to(self.device), batch_y_test.to(self.device)
                        preds_test = self.model(batch_x_test)
                        predictions_test.append(preds_test.cpu())
                        truth_test.append(batch_y_test.cpu())

                    y_val_pred_rescaled = torch.cat(predictions_test).numpy()
                    y_val_true_rescaled = torch.cat(truth_test).numpy()
                    
                    y_val_pred_original = np.squeeze(ytest_to_initial_scale(y_val_pred_rescaled, scalers['min_max_scalerY'], scalers['transformerY'], scalers['shift_value_Y']))
                    y_val_true_original = np.squeeze(ytest_to_initial_scale(y_val_true_rescaled, scalers['min_max_scalerY'], scalers['transformerY'], scalers['shift_value_Y']))

                    # Compute metrics
                    val_mape.append(mape(y_val_true_original, y_val_pred_original))
                    val_r2.append(r_squared(y_val_true_original, y_val_pred_original))

            for name in param_names:
                results_dict[name].append(params[name])
            results_dict["Val_MAPE"].append(np.mean(val_mape))
            results_dict["Val_R2"].append(np.mean(val_r2))
            results_dict["Input_features"].append(input_size)
            results_dict["Num_of_parameters"].append(num_params)
            write_csv(results_dict, hyperparameter_csv_file)
            progress_bar.progress((i + 1) / total_combinations)

        # Create and rank results
        df = pd.DataFrame(results_dict)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Val_MAPE", "Val_R2"])
        df["MAPE_rank"] = df["Val_MAPE"].rank(method="min", ascending=True)
        df["R2_rank"] = df["Val_R2"].rank(method="min", ascending=False)
        df["combined_rank"] = (df["MAPE_rank"] + df["R2_rank"]) / 2.0
        df_sorted = df.sort_values(by="combined_rank", ascending=True)

        self.best_params = df_sorted.iloc[0][param_names].to_dict()
        status.update(label="Tuning Complete! Ranking results...", state="complete")

        return df_sorted, param_names

    def get_default_param_grid(self, X_train) -> dict:
        """Return the default hyperparameter grid."""
        
        return {
            'epochs': [100],
            "batch_size": [8, 16],
            "grid_size": [4],
            "spline_order": [4],
            "lr": [0.001],
            "hidden_layer_size": f"[{X_train.shape[1] + 1}], [{2 * X_train.shape[1] + 1} {X_train.shape[1] + 1}], [{3 * X_train.shape[1] + 1} {2 * X_train.shape[1] + 1} {X_train.shape[1] + 1}]"
        }
        # "hidden_layer_size": "[10], [10 10]"
        # 'hidden_layer_size': [[X_train.shape[1]], [X_train.shape[1],X_train.shape[1]]]
        
        # return {
        #     'epochs': [100, 200, 300],
        #     "batch_size": [4, 8, 16, 32, 64, 128],
        #     "grid_size": [4, 6, 8, 10],
        #     "spline_order": [2, 3, 4, 5],
        #     "lr": [0.01, 0.05, 0.001, 0.005]
        # }
        
    #    return {
    #         'epochs': [100, 200, 300],
    #         "batch_size": [4, 8, 16, 32, 64, 128],
    #         "grid_size": [4, 6, 8, 10],
    #         "spline_order": [2, 3, 4, 5],
    #         "lr": [0.01, 0.05, 0.001, 0.005],
    #         "weight_decay": [0],
    #         "points": [1e-5, 0]
    #     }
       
    def get_param_definitions(self) -> dict:
        """Return the parameter definitions for UI configuration."""
        return {
            'epochs': {
                'label': 'Epochs of training',
                'ui_widget': 'text_list',
                'placeholder': 'e.g., 100, 200, 500',
                'type': int,
                'help': "The number of epochs used for backpropagation of erros."
            },
            'batch_size': {
                'label': 'batch_size of training',
                'ui_widget': 'text_list',
                'placeholder': 'e.g., 4, 8, 32',
                'type': int,
                'help': "batch sizes used for training."
            },
            'grid_size': {
                'label': 'grid_size of model',
                'ui_widget': 'text_list',
                'placeholder': 'e.g., 3, 4, 5',
                'type': int,
                'help': "grid_size of model."
            },
            'spline_order': {
                'label': 'spline_order of model',
                'ui_widget': 'text_list',
                'placeholder': 'e.g., 3, 4, 5',
                'type': int,
                'help': "spline_order of model."
            },
            'lr': {
                'label': 'Learning Rate',
                'ui_widget': 'text_list',
                'type': float,
                'placeholder': 'e.g., 0.001, 0.0001',
                'help': "Enter float values, separated by commas (e.g., 0.001, 0.0001)."
            },
            'hidden_layer_size': {
                'label': 'hidden_layer_size of the network',
                'ui_widget': 'text_list',
                'placeholder': 'e.g., [10], [10 20] (there is no need for , in a model but there is a need for , between different sets of params)',
                'type': str,
                'help': "hidden_layer_size of the network"
            }
        }

            # 'weight_decay': {
            #     'label': 'weight_decay of model',
            #     'ui_widget': 'text_list',
            #     'placeholder': 'e.g., 0.00001, 0',
            #     'type': float,
            #     'help': "weight_decay of model."
            # },
            # 'points': {
            #     'label': 'points of model',
            #     'ui_widget': 'text_list',
            #     'placeholder': 'e.g., 0.00001, 0',
            #     'type': float,
            #     'help': "points of model."
            # } 