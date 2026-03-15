# important packages
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

# setting directories
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents = True, exist_ok = True)

# probability clamp floor/ceiling
EPS = 1e-6

# main data pre-processing class
# to be used on any modeling technique (logistic, nn, etc.)
class Preprocessor:
    def __init__(self):
        # dealing with missing values
        self.imputer = SimpleImputer(strategy = "median")
        # scaling data
        self.scaler  = StandardScaler()
 
    def fit_transform(self, X):
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        return X
 
    def transform(self, X):
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return X
 
    def save(self, path):
        joblib.dump({"imputer": self.imputer, "scaler": self.scaler}, path)
        print(f"Preprocessor saved to {path}")
 
    def load(self, path):
        obj = joblib.load(path)
        self.imputer = obj["imputer"]
        self.scaler  = obj["scaler"]
        return self
    
# baseline model attempt, logistic regression using quantitative inputs
# SCREW WITH HYPERPARAMETERS
class LogisticBaseline:
    def __init__(self, C = 0.05):
        self.model = LogisticRegression(C = C, max_iter = 1000, solver = "lbfgs", random_state = 16)
        self.preprocessor = Preprocessor()
 
    def fit(self, X, y):
        X_proc = self.preprocessor.fit_transform(X)
        self.model.fit(X_proc, y)
        return self
 
    def predict_proba(self, X):
        X_proc = self.preprocessor.transform(X)
        raw = self.model.predict_proba(X_proc)[:, 1]
        return np.clip(raw, EPS, 1 - EPS)
 
    def save(self, tag = "M"):
        joblib.dump(self.model, MODELS_DIR / f"{tag}_logreg.pkl")
        self.preprocessor.save(MODELS_DIR / f"{tag}_logreg_prep.pkl")
 
    def load(self, tag = "M"):
        self.model = joblib.load(MODELS_DIR / f"{tag}_logreg.pkl")
        self.preprocessor.load(MODELS_DIR / f"{tag}_logreg_prep.pkl")
        return self
    
# advanced neural network attempt
# class to actually set up the neural netowrk
class _Net(nn.Module):
    """
    Feedforward neural network for binary classification.
    Uses normalization and dropout on all hiden layers, specified
    """
    def __init__(self, input_dim, hidden_dims, dropout = 0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        # appy passes and activations for each hidden layer
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h),
                       nn.BatchNorm1d(h),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.net(x).squeeze(1)
    
# inherit the neurel network model above and track results!
class MarchMadnessNet:
    """
    PyTorch wrapper: handles preprocessing, training loop, early stopping,
    and inference. Uses MPS (Apple Silicon GPU) if available, else CPU.
    """
    def __init__(self, hidden_dims = [128, 64, 32], dropout = 0.3, lr = 1e-3, weight_decay = 1e-4,
                 batch_size = 64, max_epochs = 200, patience = 20, grad_clip = 1.0):
        # setting hyperparameters and storings
        self.hidden_dims  = hidden_dims
        self.dropout = dropout
        self.lr  = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.preprocessor = Preprocessor()
        self.model = None
        self.device = self._get_device()
        self.history = {"train_loss": [], "val_loss": [], "val_brier": []}
 
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
 
    # setting up tensor for model depending on device available
    def _to_tensor(self, X, y = None):
        xt = torch.tensor(X, dtype = torch.float32).to(self.device)
        if y is not None:
            yt = torch.tensor(y, dtype=torch.float32).to(self.device)
            return xt, yt
        return xt
 
    def fit(self, X_train, y_train, X_val, y_val):
        """
        Train with early stopping on validation Brier score.
        """
        # Preprocess, same as imputing NA and standardizing/scaling data
        X_train_p = self.preprocessor.fit_transform(X_train)
        X_val_p   = self.preprocessor.transform(X_val)
 
        # fit actual model, inheriting from above
        input_dim = X_train_p.shape[1]
        self.model = _Net(input_dim, self.hidden_dims, self.dropout).to(self.device)
 
        # using ADAM optimizer and BCE loss as evaluation
        optimiser = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode = 'min', patience = 15, factor = 0.5)
        criterion = nn.BCELoss()
 
        # DataLoader for batching
        xt, yt = self._to_tensor(X_train_p, y_train)
        xv, yv = self._to_tensor(X_val_p, y_val)
        train_ds = TensorDataset(xt, yt)
        train_dl = DataLoader(train_ds, batch_size = self.batch_size, shuffle = True)
 
        best_val_brier = float("inf")
        best_weights = None
        no_improve = 0
 
        for epoch in range(1, self.max_epochs + 1):
            # training data
            self.model.train()
            train_losses = []
            for xb, yb in train_dl:
                optimiser.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip) 
                optimiser.step()
                train_losses.append(loss.item())
 
            # perform on validation
            # NEED TO FIX? ?????????
            # CHECK LATER CHECK LATER
            self.model.eval()
            with torch.no_grad():
                # val_preds = self.model(xv).cpu().numpy()
                
                # # FIX: Get logits by temporarily bypassing sigmoid
                # with torch.no_grad():
                #     # Skip final Sigmoid
                #     logits = self.model.net[:-1](xv).squeeze(1)
                #     val_loss = criterion(logits, yv).item()

                val_preds = self.model(xv)
                val_probs = torch.clamp(val_preds, EPS, 1 - EPS)
                val_loss = criterion(val_preds, yv).item()
                preds_np = val_probs.cpu().numpy()
 
            # stroing training and validation scores
            val_brier = float(np.mean((preds_np - y_val) ** 2))
            train_loss = float(np.mean(train_losses))
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_brier"].append(val_brier)
 
            scheduler.step(val_brier)
 
            # perform early stopping if needed
            if val_brier < best_val_brier - 1e-6:
                best_val_brier = val_brier
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
 
            if epoch % 20 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | train_loss = {train_loss:.4f} "
                      f"val_loss = {val_loss:.4f}  val_brier = {val_brier:.4f} "
                      f"({'best' if no_improve == 0 else f'no improve {no_improve}'})")
 
            if no_improve >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best val Brier: {best_val_brier:.4f}")
                break
 
        self.model.load_state_dict(best_weights)
        self.model.to(self.device)
        print(f"\nTraining complete. Best val Brier: {best_val_brier:.4f}")
        return self
 
    def predict_proba(self, X):
        X_p = self.preprocessor.transform(X)
        xt  = torch.tensor(X_p, dtype = torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            raw = self.model(xt)
            return torch.clamp(raw, EPS, 1 - EPS).cpu().numpy()
 
    def save(self, tag = "M"):
        torch.save(self.model.state_dict(), MODELS_DIR / f"{tag}_nn.pt")
        self.preprocessor.save(MODELS_DIR / f"{tag}_nn_prep.pkl")
        print(f"Model saved → models/{tag}_nn.pt")
 
    def load(self, tag = "M", input_dim = None):
        self.preprocessor.load(MODELS_DIR / f"{tag}_nn_prep.pkl")

        try:
            input_dim = getattr(self.preprocessor.scaler, 'n_features_in_', None)
        except:
            raise ValueError("Preprocessor must be fitted first or provide input_dim")

        self.model = _Net(input_dim, self.hidden_dims, self.dropout).to(self.device)
        self.model.load_state_dict(torch.load(MODELS_DIR / f"{tag}_nn.pt", map_location = self.device))
        self.model.eval()
        print(f"Model {tag} loaded successfully")
        return self
    
# finally, simple function to print evaluations
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = ""):
    """Print Brier score and log loss."""
    y_pred_c = np.clip(y_pred, EPS, 1 - EPS)
    brier = float(np.mean((y_pred_c - y_true) ** 2))
    ll    = log_loss(y_true, y_pred_c)
    print(f"{label:35s}  Brier={brier:.4f}  LogLoss={ll:.4f}")
    return brier, ll