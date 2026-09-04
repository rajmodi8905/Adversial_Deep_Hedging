"""
Neural Network Deep Hedger in PyTorch with Differentiable CVaR Loss and Market Frictions.
Implements a Residual Delta Architecture around Black-Scholes Greeks for fast convergence.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .black_scholes import black_scholes_greeks


class DeepHedgerNet(nn.Module):
    """
    Residual Deep Hedger:
    Predicts the optimal friction adjustment and inertia band around the analytical Black-Scholes Delta.

    Inputs per step:
        1. Normalized Moneyness: S_t / K
        2. Black-Scholes Delta: Delta_BS
        3. Remaining Time Fraction: tau = (T - t) / T
        4. Previous Delta Position: delta_{t-1}
        5. Position Discrepancy: delta_{t-1} - Delta_BS
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, option_type: str = "call"):
        super().__init__()
        self.option_type = option_type.lower()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),  # Outputs [trade_decision, target_adjustment]
        )

    def forward(self, state: torch.Tensor, bs_delta: torch.Tensor, prev_delta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic inertia gating.
        """
        out = self.net(state)
        # out[:, 0]: trade gate in [0, 1] (0 = hold prev_delta, 1 = rebalance)
        # out[:, 1]: delta adjustment [-0.2, +0.2]
        trade_gate = torch.sigmoid(out[:, 0:1])
        delta_adj = torch.tanh(out[:, 1:2]) * 0.15

        target_delta = bs_delta + delta_adj
        if self.option_type == "call":
            target_delta = torch.clamp(target_delta, 0.0, 1.0)
        else:
            target_delta = torch.clamp(target_delta, -1.0, 0.0)

        # Smooth rebalancing decision between holding prev_delta and moving to target_delta
        action_delta = prev_delta + trade_gate * (target_delta - prev_delta)
        return action_delta


class CVaRLoss(nn.Module):
    """
    Differentiable Rockafellar-Uryasev Conditional Value at Risk (CVaR) Loss.
    Minimizes the expected tail loss beyond the (1 - alpha) quantile.
    """

    def __init__(self, alpha: float = 0.05, initial_w: float = 50.0):
        super().__init__()
        self.alpha = alpha
        self.w = nn.Parameter(torch.tensor(initial_w, dtype=torch.float32))

    def forward(self, pnl: torch.Tensor) -> torch.Tensor:
        """
        pnl shape: (batch_size,)
        loss = w + (1 / alpha) * E[max(0, -pnl - w)]
        """
        loss_tail = torch.clamp(-pnl - self.w, min=0.0)
        cvar_val = self.w + (1.0 / self.alpha) * torch.mean(loss_tail)
        return cvar_val


def compute_pnl_torch(
    paths: torch.Tensor,
    deltas: torch.Tensor,
    K: float,
    cost_rate: float = 0.0,
    option_type: str = "call"
) -> torch.Tensor:
    """
    Differentiable P&L computation for PyTorch tensors.

    paths: (batch_size, num_steps + 1)
    deltas: (batch_size, num_steps + 1)
    """
    S_T = paths[:, -1]
    if option_type == "call":
        payoff = torch.clamp(S_T - K, min=0.0)
    else:
        payoff = torch.clamp(K - S_T, min=0.0)

    # Trading P&L
    price_diffs = paths[:, 1:] - paths[:, :-1]
    trading_pnl = torch.sum(deltas[:, :-1] * price_diffs, dim=1)

    # Transaction costs
    costs = torch.zeros_like(trading_pnl)
    if cost_rate > 0.0:
        costs += cost_rate * paths[:, 0] * torch.abs(deltas[:, 0])
        rebalance_diffs = torch.abs(deltas[:, 1:-1] - deltas[:, :-2])
        costs += torch.sum(cost_rate * paths[:, 1:-1] * rebalance_diffs, dim=1)
        costs += cost_rate * paths[:, -1] * torch.abs(deltas[:, -2])

    pnl = -payoff + trading_pnl - costs
    return pnl


class DeepHedger:
    """
    Wrapper for training, evaluating, and applying the Deep Hedging model.
    """

    def __init__(
        self,
        K: float,
        T: float,
        r: float,
        sigma: float,
        cost_rate: float = 0.001,
        hidden_dim: int = 32,
        option_type: str = "call",
        lr: float = 0.003,
        alpha: float = 0.05,
        device: str = "cpu"
    ):
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.cost_rate = cost_rate
        self.option_type = option_type.lower()
        self.alpha = alpha
        self.device = torch.device(device)

        self.model = DeepHedgerNet(input_dim=5, hidden_dim=hidden_dim, option_type=self.option_type).to(self.device)
        self.cvar_loss_fn = CVaRLoss(alpha=self.alpha, initial_w=50.0).to(self.device)
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.cvar_loss_fn.parameters()),
            lr=lr
        )

    def _prepare_bs_deltas(self, paths_np: np.ndarray) -> np.ndarray:
        num_paths, total_steps = paths_np.shape
        num_steps = total_steps - 1
        dt = self.T / num_steps
        bs_deltas = np.zeros_like(paths_np)
        for s in range(total_steps):
            t_rem = max(1e-7, self.T - s * dt)
            greeks = black_scholes_greeks(paths_np[:, s], self.K, t_rem, self.r, self.sigma, option_type=self.option_type)
            bs_deltas[:, s] = greeks["delta"]
        return bs_deltas

    def predict_deltas(self, paths_np: np.ndarray) -> np.ndarray:
        """
        Runs neural hedger over an array of paths in numpy and outputs deltas.
        """
        self.model.eval()
        num_paths, total_steps = paths_np.shape
        num_steps = total_steps - 1
        dt = self.T / num_steps

        bs_deltas_np = self._prepare_bs_deltas(paths_np)

        paths_tensor = torch.tensor(paths_np, dtype=torch.float32, device=self.device)
        bs_deltas_tensor = torch.tensor(bs_deltas_np, dtype=torch.float32, device=self.device)

        all_deltas = []
        prev_delta = bs_deltas_tensor[:, 0:1]  # Initialize at t=0 BS delta

        with torch.no_grad():
            for t in range(total_steps):
                time_frac = (self.T - t * dt) / self.T
                S_t = paths_tensor[:, t : t + 1]
                bs_d_t = bs_deltas_tensor[:, t : t + 1]
                norm_moneyness = S_t / self.K
                time_tensor = torch.full_like(norm_moneyness, time_frac)
                delta_gap = prev_delta - bs_d_t

                state = torch.cat([norm_moneyness, bs_d_t, time_tensor, prev_delta, delta_gap], dim=-1)
                pred_delta = self.model(state, bs_d_t, prev_delta)
                all_deltas.append(pred_delta)
                prev_delta = pred_delta

            stacked_deltas = torch.cat(all_deltas, dim=1).cpu().numpy()

        return stacked_deltas

    def fit(self, train_paths: np.ndarray, epochs: int = 35, batch_size: int = 256, verbose: bool = True):
        """
        Trains the deep hedger model on a set of simulated price paths.
        """
        self.model.train()
        num_paths, total_steps = train_paths.shape
        num_steps = total_steps - 1
        dt = self.T / num_steps

        bs_deltas_np = self._prepare_bs_deltas(train_paths)

        paths_tensor = torch.tensor(train_paths, dtype=torch.float32, device=self.device)
        bs_deltas_tensor = torch.tensor(bs_deltas_np, dtype=torch.float32, device=self.device)

        num_batches = int(np.ceil(num_paths / batch_size))
        loss_history = []

        for epoch in range(epochs):
            perm = torch.randperm(num_paths)
            epoch_loss = 0.0

            for b in range(num_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                b_paths = paths_tensor[idx]
                b_bs_deltas = bs_deltas_tensor[idx]
                curr_b_size = len(idx)

                deltas_list = []
                prev_delta = b_bs_deltas[:, 0:1]

                for t in range(total_steps):
                    time_frac = (self.T - t * dt) / self.T
                    S_t = b_paths[:, t : t + 1]
                    bs_d_t = b_bs_deltas[:, t : t + 1]
                    norm_moneyness = S_t / self.K
                    time_tensor = torch.full_like(norm_moneyness, time_frac)
                    delta_gap = prev_delta - bs_d_t

                    state = torch.cat([norm_moneyness, bs_d_t, time_tensor, prev_delta, delta_gap], dim=-1)
                    pred_delta = self.model(state, bs_d_t, prev_delta)
                    deltas_list.append(pred_delta)
                    prev_delta = pred_delta

                batch_deltas = torch.cat(deltas_list, dim=1)

                pnl = compute_pnl_torch(
                    b_paths,
                    batch_deltas,
                    self.K,
                    cost_rate=self.cost_rate,
                    option_type=self.option_type
                )

                loss = self.cvar_loss_fn(pnl)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * curr_b_size

            avg_loss = epoch_loss / num_paths
            loss_history.append(avg_loss)

            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1):
                print(f"Epoch [{epoch + 1:02d}/{epochs:02d}] | CVaR Loss (alpha={self.alpha:.0%}): {avg_loss:.2f} | VaR threshold: {self.cvar_loss_fn.w.item():.2f}")

        return loss_history
