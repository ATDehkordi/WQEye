# File: src/models/rnn.py
import os
import csv
import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import KFold, ParameterGrid

from src.models.base_model import BaseMLModel
from src.utils.scaling_utils import mape, r_squared, ytest_to_initial_scale


# ----------------------------
# Your original RNN architecture (logic preserved)
# ----------------------------
class RNNRegressor(nn.Module):
    # def __init__(self, bidirectional, input_RNN_size, RNN_layers_with_hidden_sizes, FCNN_layers_with_hidden_sizes, FCNN_act, FCNN_dropout_rate):
    def __init__(
        self,
        bidirectional,
        input_RNN_size,
        RNN_layers_with_hidden_sizes,
        FCNN_layers_with_hidden_sizes,
        FCNN_act,
        FCNN_dropout_rate,
        number_out_parameters,
    ):
        super().__init__()
        """
        If FCNN_layers_with_hidden_sizes is [0], no FCNN head is applied and the network passes only
        the last hidden state to a final Linear layer. 'bidirectional' must be True or False.
        """
        self.bidirectional = bidirectional
        self.input_RNN_size = input_RNN_size
        self.RNN_layers_with_hidden_sizes = RNN_layers_with_hidden_sizes

        # FCNN input size depends on bidirectionality
        self.input_FCNN_size = (
            RNN_layers_with_hidden_sizes[-1] * 2 if self.bidirectional else RNN_layers_with_hidden_sizes[-1]
        )
        self.FCNN_layers_with_hidden_sizes = FCNN_layers_with_hidden_sizes
        self.FCNN_act = FCNN_act
        self.FCNN_dropout_rate = FCNN_dropout_rate
        self.number_out_parameters = number_out_parameters

        # ---- RNN stack ----
        layers_RNN = []
        in_RNN_size = self.input_RNN_size
        for hidden_size in self.RNN_layers_with_hidden_sizes:
            layers_RNN.append(
                nn.RNN(
                    input_size=in_RNN_size,
                    hidden_size=hidden_size,
                    bias=True,
                    batch_first=True,
                    bidirectional=self.bidirectional,
                )
            )
            in_RNN_size = hidden_size * 2 if self.bidirectional else hidden_size
        self.layers_RNN = nn.ModuleList(layers_RNN)

        # ---- FCNN head ----
        if len(FCNN_layers_with_hidden_sizes) == 1 and FCNN_layers_with_hidden_sizes[0] == 0:
            # No hidden FC layers; direct linear to outputs
            self.layers_FCNN = nn.ModuleList(
                [nn.Linear(self.input_FCNN_size, self.number_out_parameters, bias=True)]
            )
        else:
            layers_FCNN = []
            in_FCNN_size = self.input_FCNN_size

            for hidden_layer in self.FCNN_layers_with_hidden_sizes:
                layers_FCNN.append(nn.Linear(in_FCNN_size, hidden_layer, bias=True))
                in_FCNN_size = hidden_layer

                if self.FCNN_act == "relu":
                    layers_FCNN.append(nn.ReLU())
                elif self.FCNN_act == "sigmoid":
                    layers_FCNN.append(nn.Sigmoid())
                elif self.FCNN_act == "tanh":
                    layers_FCNN.append(nn.Tanh())

                if self.FCNN_dropout_rate > 0:
                    layers_FCNN.append(nn.Dropout(self.FCNN_dropout_rate))

            layers_FCNN.append(nn.Linear(self.FCNN_layers_with_hidden_sizes[-1], self.number_out_parameters, bias=True))
            self.layers_FCNN = nn.ModuleList(layers_FCNN)

    def forward(self, x):
        # Pass through stacked vanilla RNNs
        for rnn in self.layers_RNN:
            x, hn = rnn(x)  # hn: (num_layers * num_directions, B, H)

        if self.bidirectional:
            # Concatenate last forward/backward hidden states
            final_fw = hn[0, :, :]  # (B, H)
            final_bw = hn[1, :, :]  # (B, H)
            x_FCNN = torch.cat([final_fw, final_bw], dim=1)
        else:
            # Last layer's hidden state
            x_FCNN = hn[-1]  # (B, H)

        # FCNN head
        for FCNN in self.layers_FCNN:
            x_FCNN = FCNN(x_FCNN)

        return x_FCNN


# ----------------------------
# Project-aligned wrapper model (same pattern as LSTM/GRU)
# ----------------------------
class RNNModel(BaseMLModel):
    """Vanilla RNN model aligned with the project's training/tuning/predict interfaces."""

    def __init__(self):
        self.model = None
        self.best_params = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "Recurrent Neural Network (RNN)"

    # ----- helpers -----
    def _ensure_3d(self, X_df: pd.DataFrame) -> torch.Tensor:
        """
        Convert tabular input (N, T) into (N, T, 1) for RNN input.
        Each DataFrame column is treated as a time step; a singleton feature dimension is appended.
        """
        X_np = X_df.to_numpy().astype(np.float32)  # (N, T)
        X_np = X_np.reshape(X_np.shape[0], X_np.shape[1], 1)  # (N, T, 1)
        return torch.from_numpy(X_np)

    # ----- fit -----
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> None:
        """
        Final training using best params (from tune) or provided params.
        Training loop/optimizer/loss strictly follows your original logic.
        """
        params = kwargs.get("params", self.best_params if self.best_params else {})

        # Save best params CSV (consistent with other models)
        best_csv = "outputs/RNN_bestparams.csv"
        os.makedirs(os.path.dirname(best_csv), exist_ok=True)
        if os.path.isfile(best_csv):
            os.remove(best_csv)

        def _write_csv_row(d: dict, path: str):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(list(d.keys()))
                writer.writerow(list(d.values()))

        _write_csv_row(params, best_csv)

        # Data -> 3D tensors
        x_train_tensor = self._ensure_3d(X_train)
        y_train_tensor = torch.tensor(y_train.to_numpy().astype(np.float32)).reshape(-1, 1)

        train_ds = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_ds, batch_size=int(params.get("batch_size", 8)), shuffle=True)

        # Build model exactly per your logic
        self.model = RNNRegressor(
            bidirectional=bool(params.get("bidirectional", True)),
            input_RNN_size=1,
            RNN_layers_with_hidden_sizes=params.get("RNN_layers_with_hidden_sizes", [64, 64]),
            FCNN_layers_with_hidden_sizes=params.get("FCNN_layers_with_hidden_sizes", [0]),
            FCNN_act=params.get("FCNN_act", "relu"),
            FCNN_dropout_rate=float(params.get("FCNN_dropout_rate", 0.0)),
            number_out_parameters=1,
        ).to(self.device)
        
        print(f"[Device check] Model device: {next(self.model.parameters()).device}")
        if torch.cuda.is_available():
            print(f"[CUDA] Device count: {torch.cuda.device_count()}, Current: {torch.cuda.current_device()}")
            print(f"[CUDA] Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=float(params.get("lr", 1e-3)))

        epochs = int(params.get("epochs", 100))
        self.model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                preds = self.model(xb)
                loss = criterion(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # ----- predict -----
    def predict(self, X: pd.DataFrame, **kwargs) -> pd.Series:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        self.model.eval()
        with torch.no_grad():
            x_tensor = self._ensure_3d(X).to(self.device)
            preds = self.model(x_tensor).detach().cpu().numpy().reshape(-1)
        return pd.Series(preds, index=X.index)

    # ----- tune -----
    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid: dict, k_folds: int, scalers: dict, **kwargs):
        """
        KFold + GridSearch with Streamlit progress, CSV logging, and ranking,
        fully consistent with the LSTM/GRU integrations.
        """
        hyper_csv = "outputs/RNN_hyperparametertuning.csv"
        os.makedirs(os.path.dirname(hyper_csv), exist_ok=True)
        if os.path.isfile(hyper_csv):
            os.remove(hyper_csv)

        # Normalize possible string-based layer lists (e.g., "[64 64], [32]" -> [[64,64],[32]])
        def _normalize_layers_list(value):
            if len(value) == 0:
                return value
            if isinstance(value[0], str):
                parsed = []
                for item in value:
                    nums = [int(x) for x in item.strip().strip("[]").split()]
                    parsed.append(nums)
                return parsed
            return value

        param_grid = dict(param_grid)
        param_grid["RNN_layers_with_hidden_sizes"] = _normalize_layers_list(
            param_grid.get("RNN_layers_with_hidden_sizes", [[64, 64], [32]])
        )
        param_grid["FCNN_layers_with_hidden_sizes"] = _normalize_layers_list(
            param_grid.get("FCNN_layers_with_hidden_sizes", [[0], [16, 16], [16], [8]])
        )

        param_names = list(param_grid.keys())
        results = {key: [] for key in param_names}
        results.update({"Val_MAPE": [], "Val_R2": [], "Input_features": [], "Num_of_parameters": []})

        combos = list(ParameterGrid(param_grid))
        total = len(combos)
        st.write(f"Total parameter combinations: **{total}**")
        status = st.status("Starting RNN tuning...", expanded=False)
        progress = st.progress(0)

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        def _write_full(results_dict: dict, path: str):
            headers = list(results_dict.keys())
            rows = list(zip(*results_dict.values()))
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)

        for i, params in enumerate(combos):
            status.update(label=f"Testing Combination {i+1}/{total}")

            fold_mape, fold_r2 = [], []
            input_size = X_train.shape[1]
            num_params_last = None

            for train_idx, val_idx in kf.split(X_train):
                X_tr = X_train.iloc[train_idx]
                y_tr = y_train.iloc[train_idx]
                X_val = X_train.iloc[val_idx]
                y_val = y_train.iloc[val_idx]

                x_tr_tensor = self._ensure_3d(X_tr)
                y_tr_tensor = torch.tensor(y_tr.to_numpy().astype(np.float32)).reshape(-1, 1)
                x_val_tensor = self._ensure_3d(X_val)
                y_val_tensor = torch.tensor(y_val.to_numpy().astype(np.float32)).reshape(-1, 1)

                train_ds = TensorDataset(x_tr_tensor, y_tr_tensor)
                val_ds = TensorDataset(x_val_tensor, y_val_tensor)

                train_loader = DataLoader(train_ds, batch_size=int(params.get("batch_size")), shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=int(params.get("batch_size")), shuffle=False)

                model = RNNRegressor(
                    bidirectional=bool(params.get("bidirectional")),
                    input_RNN_size=1,
                    RNN_layers_with_hidden_sizes=params.get("RNN_layers_with_hidden_sizes"),
                    FCNN_layers_with_hidden_sizes=params.get("FCNN_layers_with_hidden_sizes"),
                    FCNN_act=params.get("FCNN_act"),
                    FCNN_dropout_rate=float(params.get("FCNN_dropout_rate")),
                    number_out_parameters=1,
                ).to(self.device)

                num_params_last = sum(p.numel() for p in model.parameters() if p.requires_grad)

                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=float(params.get("lr")))
                epochs = int(params.get("epochs"))

                model.train()
                for _ in range(epochs):
                    for xb, yb in train_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        preds = model(xb)
                        loss = criterion(preds, yb)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    y_true_list, y_pred_list = [], []
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        pb = model(xb)
                        y_true_list.append(yb.detach().cpu())
                        y_pred_list.append(pb.detach().cpu())

                    y_val_true = torch.cat(y_true_list, dim=0).numpy()
                    y_val_pred = torch.cat(y_pred_list, dim=0).numpy()

                    # Return to the original scale (project-consistent)
                    y_val_pred_orig = np.squeeze(
                        ytest_to_initial_scale(
                            y_val_pred, scalers["min_max_scalerY"], scalers["transformerY"], scalers["shift_value_Y"]
                        )
                    )
                    y_val_true_orig = np.squeeze(
                        ytest_to_initial_scale(
                            y_val_true, scalers["min_max_scalerY"], scalers["transformerY"], scalers["shift_value_Y"]
                        )
                    )

                    fold_mape.append(mape(y_val_true_orig, y_val_pred_orig))
                    fold_r2.append(r_squared(y_val_true_orig, y_val_pred_orig))

            # Log results for this combination
            for name in param_names:
                results[name].append(params[name])
            results["Val_MAPE"].append(np.mean(fold_mape))
            results["Val_R2"].append(np.mean(fold_r2))
            results["Input_features"].append(input_size)
            results["Num_of_parameters"].append(num_params_last)
            _write_full(results, hyper_csv)
            progress.progress((i + 1) / total)

        # Ranking & store best
        df = pd.DataFrame(results)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Val_MAPE", "Val_R2"])
        df["MAPE_rank"] = df["Val_MAPE"].rank(method="min", ascending=True)
        df["R2_rank"] = df["Val_R2"].rank(method="min", ascending=False)
        df["combined_rank"] = (df["MAPE_rank"] + df["R2_rank"]) / 2.0
        df_sorted = df.sort_values(by="combined_rank", ascending=True)

        self.best_params = df_sorted.iloc[0][param_names].to_dict()
        status.update(label="RNN tuning complete!", state="complete")

        return df_sorted, param_names

    # ----- default grid -----
    def get_default_param_grid(self, X_train) -> dict:
        """
        EXACT default (mirrors your previous defaults, keyed for RNN).
        """
        return {
            "bidirectional": [False],
            "RNN_layers_with_hidden_sizes": [[64, 64]],
            "FCNN_layers_with_hidden_sizes": [[16, 16]],
            "FCNN_act": ["relu"],
            "FCNN_dropout_rate": [0.0],
            "lr": [5e-4],
            "batch_size": [16],
            "epochs": [100],
        }

    # ----- param definitions -----
    def get_param_definitions(self) -> dict:
        """
        EXACTLY these parameters are exposed (all comments in English).
        """
        return {
            "bidirectional": {
                "label": "Bidirectional RNN",
                "ui_widget": "multiselect",
                "options": [True, False],
                "type": bool,
                "help": "Use bidirectional RNNs or not.",
            },
            "RNN_layers_with_hidden_sizes": {
                "label": "RNN Hidden Sizes (per layer)",
                "ui_widget": "text_list",
                "placeholder": "e.g., [64 64], [32 32], [64 32], [64], [32]",
                "type": str,
                "help": "List(s) of hidden sizes per RNN layer. Use bracketed lists separated by commas.",
            },
            "FCNN_layers_with_hidden_sizes": {
                "label": "FCNN Hidden Sizes",
                "ui_widget": "text_list",
                "placeholder": "e.g., [0] or [16 16], [16 8], [16], [8]",
                "type": str,
                "help": "Set [0] to disable the FCNN head; otherwise provide bracketed hidden sizes.",
            },
            "FCNN_act": {
                "label": "FCNN Activation",
                "ui_widget": "multiselect",
                "options": ["relu", "sigmoid", "tanh"],
                "type": str,
                "help": "Activation function used between FCNN layers.",
            },
            "FCNN_dropout_rate": {
                "label": "FCNN Dropout Rate",
                "ui_widget": "text_list",
                "placeholder": "e.g., 0.0",
                "type": float,
                "help": "Dropout probability after FCNN hidden layers (0 disables dropout).",
            },
            "lr": {
                "label": "Learning Rate",
                "ui_widget": "text_list",
                "placeholder": "e.g., 0.0005, 0.001, 0.005",
                "type": float,
                "help": "Adam learning rate.",
            },
            "batch_size": {
                "label": "Batch Size",
                "ui_widget": "text_list",
                "placeholder": "e.g., 4, 8, 16, 32, 64",
                "type": int,
                "help": "Batch size for training.",
            },
            "epochs": {
                "label": "Epochs",
                "ui_widget": "text_list",
                "placeholder": "e.g., 100",
                "type": int,
                "help": "Number of training epochs.",
            },
        }
