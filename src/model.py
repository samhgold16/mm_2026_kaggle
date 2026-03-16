import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
import joblib
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple
import warnings
import xgboost as xgb

# setting directories
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents = True, exist_ok = True)

# probability clamp floor/ceiling
EPS = 1e-7

# main data pre-processing class
# to be used on any modeling technique (logistic, nn, etc.)
class Preprocessor:
    def __init__(self, strategy="median"):
        # dealing with missing values
        self.imputer = SimpleImputer(strategy = strategy)
        # scaling data
        self.scaler  = StandardScaler()
        self.feature_names = None
 
    def fit_transform(self, X, feature_names=None):
        if feature_names is not None:
            self.feature_names = feature_names
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        return X
 
    def transform(self, X):
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return X
 
    def save(self, path):
        joblib.dump({ "imputer": self.imputer, "scaler": self.scaler, "feature_names": self.feature_names}, path)
        print(f"Preprocessor saved to {path}")
 
    def load(self, path):
        obj = joblib.load(path)
        self.imputer = obj["imputer"]
        self.scaler  = obj["scaler"]
        self.feature_names = obj.get("feature_names", None)
        return self
    
# baseline model attempt, logistic regression using quantitative inputs
# SCREW WITH HYPERPARAMETERS
class LogisticBaseline:
    def __init__(self, C = 0.05):
        self.model = LogisticRegression(C = C, max_iter = 2000, solver = "lbfgs", random_state = 16)
        self.preprocessor = Preprocessor()
 
    def fit(self, X, y, feature_names=None):
        X_proc = self.preprocessor.fit_transform(X, feature_names)
        self.model.fit(X_proc, y)
        return self
 
    def predict_proba(self, X):
        X_proc = self.preprocessor.transform(X)
        raw = self.model.predict_proba(X_proc)[:, 1]
        return np.clip(raw, EPS, 1 - EPS)
 
    def get_feature_importance(self):
        """Return feature coefficients for interpretability."""
        if self.preprocessor.feature_names is None:
            return None
        return dict(zip(self.preprocessor.feature_names, self.model.coef_[0]))

    def save(self, tag="M"):
        joblib.dump(self.model, MODELS_DIR / f"{tag}_logreg.pkl")
        self.preprocessor.save(MODELS_DIR / f"{tag}_logreg_prep.pkl")

    def load(self, tag="M"):
        self.model = joblib.load(MODELS_DIR / f"{tag}_logreg.pkl")
        self.preprocessor.load(MODELS_DIR / f"{tag}_logreg_prep.pkl")
        return self
    

# XGBoost Baseline (for ensemble)
class XGBBaseline:
    """
    XGBoost classifier 
    """
    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                 reg_alpha=0.1, random_state=16):

        self.params = {"n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "subsample": subsample,
                        "colsample_bytree": colsample_bytree,
                        "reg_lambda": reg_lambda,
                        "reg_alpha": reg_alpha,
                        "random_state": random_state,
                        "objective": "binary:logistic",
                        "eval_metric": "logloss",
                        "use_label_encoder": False}
        self.model = None
        self.preprocessor = Preprocessor()

    def fit(self, X, y, X_val=None, y_val=None, feature_names=None,
            early_stopping_rounds=20):
        X_proc = self.preprocessor.fit_transform(X, feature_names)

        self.model = xgb.XGBClassifier(**self.params)

        if X_val is not None and y_val is not None:
            X_val_proc = self.preprocessor.transform(X_val)
            self.model.fit(
                X_proc, y,
                eval_set=[(X_val_proc, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_proc, y)

        return self

    def predict_proba(self, X):
        X_proc = self.preprocessor.transform(X)
        raw = self.model.predict_proba(X_proc)[:, 1]
        return np.clip(raw, EPS, 1 - EPS)

    def get_feature_importance(self):
        """Return feature importance scores."""
        if self.preprocessor.feature_names is None:
            return None
        return dict(zip(self.preprocessor.feature_names,
                        self.model.feature_importances_))

    def save(self, tag="M"):
        joblib.dump(self.model, MODELS_DIR / f"{tag}_xgb.pkl")
        self.preprocessor.save(MODELS_DIR / f"{tag}_xgb_prep.pkl")

    def load(self, tag="M"):
        self.model = joblib.load(MODELS_DIR / f"{tag}_xgb.pkl")
        self.preprocessor.load(MODELS_DIR / f"{tag}_xgb_prep.pkl")
        return self
    
# neural network architecture
class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    """
    def __init__(self, dim, dropout = 0.2, activation = "relu"):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.1)
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual  # Skip connection
        out = self.act(out)
        return out
    
# inherit the neurel network model above and track results!
class MarchMadnessNet(nn.Module):
    """
    PyTorch wrapper: handles preprocessing, training loop, early stopping,
    and inference. Uses MPS (Apple Silicon GPU) if available, else CPU.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64, 32],
                 dropout=0.3, activation="relu", use_residual=False):
        super().__init__()

        self.use_residual = use_residual

        # Select activation
        if activation == "gelu":
            act_fn = nn.GELU
        elif activation == "leaky_relu":
            act_fn = lambda: nn.LeakyReLU(0.1)
        else:
            act_fn = nn.ReLU

        # Input projection
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), act_fn(), nn.Dropout(dropout)]

        if use_residual:
            # Residual blocks (all same dimension)
            res_dim = hidden_dims[0]
            for _ in range(len(hidden_dims) - 1):
                layers.append(ResidualBlock(res_dim, dropout, activation))
            # Final output
            layers.extend([nn.Linear(res_dim, 1), nn.Sigmoid()])
        else:
            # Standard MLP with decreasing hidden dims
            in_dim = hidden_dims[0]
            for h in hidden_dims[1:]:
                layers.extend([nn.Linear(in_dim, h), nn.BatchNorm1d(h), act_fn(), nn.Dropout(dropout)])
                in_dim = h
            layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])

        self.net = nn.Sequential(*layers)

        # Temperature for calibration (learned post-training)
        self.trunk = nn.Sequential(*layers)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)
 
    def forward(self, x):
        return self.net(x).squeeze(1)

    def forward_with_temperature(self, x):
        """Apply temperature scaling for calibration."""
        logits = self.trunk(x).squeeze(1)
        return torch.sigmoid(logits / self.temperature)
    
# training wrapper for network
class MarchMadnessTrainer:
    def __init__(self, hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.3, activation: str = "relu",
                 use_residual: bool = False, lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 64,
                 max_epochs: int = 200, patience: int = 25, grad_clip: float = 1.0, label_smoothing: float = 0.0,
                 lr_scheduler: str = "plateau"):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation
        self.use_residual = use_residual
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.label_smoothing = label_smoothing
        self.lr_scheduler_type = lr_scheduler

        self.preprocessor = Preprocessor()
        self.model = None
        self.device = self._get_device()
        self.history = {"train_loss": [], "val_loss": [], "val_brier": [], "lr": []}

    @staticmethod
    def _get_device():
        if torch.backends.mps.is_available():
            print("Using Apple MPS (GPU)")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("Using CUDA GPU")
            return torch.device("cuda")
        else:
            print("Using CPU")
            return torch.device("cpu")

    def _to_tensor(self, X, y=None):
        xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        if y is not None:
            yt = torch.tensor(y, dtype=torch.float32).to(self.device)
            return xt, yt
        return xt

    def _smooth_labels(self, y, smoothing=0.0):
        """Apply label smoothing: y -> y*(1-s) + 0.5*s"""
        if smoothing > 0:
            return y * (1 - smoothing) + 0.5 * smoothing
        return y
    
    def fit(self, X_train, y_train, X_val, y_val, feature_names=None):
        """Train with early stopping on validation Brier score."""

        # Preprocess
        X_train_p = self.preprocessor.fit_transform(X_train, feature_names)
        X_val_p = self.preprocessor.transform(X_val)

        input_dim = X_train_p.shape[1]
        print(f"\nInput dimension: {input_dim}")
        print(f"Architecture: {self.hidden_dims}")
        print(f"Residual: {self.use_residual}, Activation: {self.activation}")

        # Initialize model
        self.model = MarchMadnessNet(input_dim, self.hidden_dims, self.dropout, self.activation,  self.use_residual).to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)

        # LR Scheduler
        if self.lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

        criterion = nn.BCELoss()

        # DataLoader
        xt, yt = self._to_tensor(X_train_p, y_train)
        xv, yv = self._to_tensor(X_val_p, y_val)

        # Apply label smoothing to training labels
        yt_smooth = self._smooth_labels(yt, self.label_smoothing)

        train_ds = TensorDataset(xt, yt_smooth)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        best_val_brier = float("inf")
        best_weights = None
        no_improve = 0

        print(f"\nStarting training (max {self.max_epochs} epochs, patience={self.patience})...")
        print("-" * 70)

        for epoch in range(1, self.max_epochs + 1):
            # Training
            self.model.train()
            train_losses = []
            for xb, yb in train_dl:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(xv)
                val_probs = torch.clamp(val_preds, EPS, 1 - EPS)
                val_loss = criterion(val_preds, yv).item()
                preds_np = val_probs.cpu().numpy()

            val_brier = float(np.mean((preds_np - y_val) ** 2))
            train_loss = float(np.mean(train_losses))
            current_lr = optimizer.param_groups[0]['lr']

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_brier"].append(val_brier)
            self.history["lr"].append(current_lr)

            # Update scheduler
            if self.lr_scheduler_type == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_brier)

            # Early stopping check
            if val_brier < best_val_brier - 1e-6:
                best_val_brier = val_brier
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} "
                      f"brier={val_brier:.4f} lr={current_lr:.2e} "
                      f"({'*best*' if no_improve == 0 else f'wait {no_improve}'})")

            if no_improve >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}. Best Brier: {best_val_brier:.4f}")
                break

        # Restore best weights
        self.model.load_state_dict(best_weights)
        self.model.to(self.device)
        print(f"\nTraining complete. Best validation Brier: {best_val_brier:.4f}")

        return self
    
    def calibrate_temperature(self, X_val, y_val, lr=0.01, max_iter=100):
        """
        Learn temperature scaling on validation set for better calibration.
        """
        print("\nCalibrating temperature...")
        X_val_p = self.preprocessor.transform(X_val)
        xv = self._to_tensor(X_val_p)
        yv = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        # Enable gradient for temperature
        self.model.temperature.requires_grad = True
        optimizer = torch.optim.LBFGS([self.model.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            with torch.no_grad():
                logits = self.model.net[:-1](xv).squeeze(1)
            scaled_probs = torch.sigmoid(logits / self.model.temperature)
            loss = nn.BCELoss()(scaled_probs, yv)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.model.temperature.requires_grad = False

        print(f"Learned temperature: {self.model.temperature.item():.4f}")
        return self
    
    def predict_proba(self, X, use_temperature=False):
        """Generate predictions."""
        X_p = self.preprocessor.transform(X)
        xt = torch.tensor(X_p, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            if use_temperature:
                raw = self.model.forward_with_temperature(xt)
            else:
                raw = self.model(xt)
            return torch.clamp(raw, EPS, 1 - EPS).cpu().numpy()

    def save(self, tag="M"):
        torch.save(self.model.state_dict(), MODELS_DIR / f"{tag}_nn.pt")
        self.preprocessor.save(MODELS_DIR / f"{tag}_nn_prep.pkl")
        # Save config
        config = {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "activation": self.activation,
            "use_residual": self.use_residual,
        }
        with open(MODELS_DIR / f"{tag}_nn_config.json", "w") as f:
            json.dump(config, f)
        print(f"Model saved -> models/{tag}_nn.pt")

    def load(self, tag="M"):
        self.preprocessor.load(MODELS_DIR / f"{tag}_nn_prep.pkl")

        # Load config
        config_path = MODELS_DIR / f"{tag}_nn_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.hidden_dims = config.get("hidden_dims", self.hidden_dims)
            self.dropout = config.get("dropout", self.dropout)
            self.activation = config.get("activation", self.activation)
            self.use_residual = config.get("use_residual", self.use_residual)

        input_dim = self.preprocessor.scaler.n_features_in_
        self.model = MarchMadnessNet(
            input_dim, self.hidden_dims, self.dropout, self.activation, self.use_residual
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(MODELS_DIR / f"{tag}_nn.pt", map_location=self.device)
        )
        self.model.eval()
        print(f"Model {tag} loaded successfully")
        return self
    
# ensemble for overall calibrating
class EnsemblePredictor:
    """
    Simple weighted ensemble of multiple models.
    """
    def __init__(self, models: List, weights: List[float] = None):
        self.models = models
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = np.array(weights) / np.sum(weights)

    def predict_proba(self, X):
        preds = np.zeros(len(X))
        for model, weight in zip(self.models, self.weights):
            preds += weight * model.predict_proba(X)
        return np.clip(preds, EPS, 1 - EPS)
    
# helper function for grid searching

def manual_grid_search(X_train, y_train, X_val, y_val,
                       param_grid: Dict,feature_names=None) -> Tuple[Dict, float, "MarchMadnessTrainer"]:
    from itertools import product

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combos = list(product(*values))

    print(f"Grid search over {len(all_combos)} combinations...")
    print("=" * 70)

    results = []
    best_brier = float("inf")
    best_params = None
    best_model = None

    for i, combo in enumerate(all_combos):
        params = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(all_combos)}] Testing: {params}")

        trainer = MarchMadnessTrainer(**params, max_epochs=150, patience=20)
        trainer.fit(X_train, y_train, X_val, y_val, feature_names)

        val_preds = trainer.predict_proba(X_val)
        brier = float(np.mean((val_preds - y_val) ** 2))

        results.append({"params": params, "brier": brier})
        print(f"    -> Brier: {brier:.4f}")

        if brier < best_brier:
            best_brier = brier
            best_params = params
            best_model = trainer

    print("\n" + "=" * 70)
    print(f"Best params: {best_params}")
    print(f"Best Brier: {best_brier:.4f}")

    return best_params, best_brier, best_model

# evaluation utilities
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> Tuple[float, float]:
    """Compute and print Brier score and log loss."""
    y_pred_c = np.clip(y_pred, EPS, 1 - EPS)
    brier = float(np.mean((y_pred_c - y_true) ** 2))
    ll = log_loss(y_true, y_pred_c)
    print(f"{label:40s}  Brier={brier:.4f}  LogLoss={ll:.4f}")
    return brier, ll


def compute_calibration(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10):
    """
    Compute calibration curve data for plotting.
    Returns (fraction_of_positives, mean_predicted_value)
    """
    fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    return fraction_pos, mean_pred


def compute_expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10):
    """
    Compute Expected Calibration Error (ECE).
    Lower is better (0 = perfectly calibrated).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred[mask].mean()
            ece += mask.sum() * np.abs(bin_acc - bin_conf)

    return ece / len(y_true)