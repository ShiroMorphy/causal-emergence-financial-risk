"""
Stiefel Manifold Optimizer for Optimal Causal Coarse-Graining
===========================================================
Finds orthogonal projection matrix W in V_q(R^p) that maximizes macro Effective Information
using exact conditional Markov projections and scale-invariant interventions.
"""

from typing import Tuple, Optional
import numpy as np

try:
    import torch

    def optimize_stiefel_torch(
        A_np: np.ndarray,
        Sigma_eps_np: np.ndarray,
        q: int,
        Sigma_x_np: Optional[np.ndarray] = None,
        kappa_do: float = 1.0,
        n_restarts: int = 12,
        max_iter: int = 100,
        lr: float = 0.05
    ) -> Tuple[np.ndarray, float]:
        p = A_np.shape[0]
        device = torch.device("cpu")
        A = torch.tensor(A_np, dtype=torch.float64, device=device)
        Sigma_eps = torch.tensor(Sigma_eps_np, dtype=torch.float64, device=device)
        
        if Sigma_x_np is not None:
            Sigma_x = torch.tensor(Sigma_x_np, dtype=torch.float64, device=device)
            # Reference total variance for PCA initialization
            init_cov = Sigma_x_np
        else:
            Sigma_x = None
            init_cov = Sigma_eps_np

        I_q = torch.eye(q, dtype=torch.float64, device=device)

        best_W = None
        best_ei = -float("inf")

        # PCA initialization on top q eigenvectors of covariance
        eigvals, eigvecs = np.linalg.eigh(init_cov)
        top_indices = np.argsort(eigvals)[::-1][:q]
        W_pca = eigvecs[:, top_indices].T

        for restart in range(n_restarts):
            if restart == 0:
                W = torch.tensor(W_pca, dtype=torch.float64, device=device, requires_grad=True)
            else:
                Z = torch.randn(p, q, dtype=torch.float64, device=device)
                Q, _ = torch.linalg.qr(Z)
                W = Q.t().detach().clone().requires_grad_(True)

            current_lr = lr
            for _ in range(max_iter):
                # Projected covariances
                if Sigma_x is not None:
                    Sigma_x_y = W @ Sigma_x @ W.t()
                    var_scale_x = torch.trace(Sigma_x_y) / float(q)
                    jitter_x = 1e-8 * var_scale_x * I_q
                    Sigma_x_y = 0.5 * (Sigma_x_y + Sigma_x_y.t()) + jitter_x

                    L_x = torch.linalg.cholesky(Sigma_x_y)
                    WASigmaW = W @ A @ Sigma_x @ W.t()
                    A_y = torch.linalg.solve(L_x.t(), torch.linalg.solve(L_x, WASigmaW.t()).t())
                    var_scale = var_scale_x
                else:
                    A_y = W @ A @ W.t()
                    var_scale = torch.trace(W @ Sigma_eps @ W.t()) / float(q)

                var_scale = torch.clamp(var_scale, min=1e-12)
                eff_sigma_do_sq = (kappa_do ** 2) * var_scale

                # Noise covariance
                Sigma_eps_y = W @ Sigma_eps @ W.t()
                jitter_eps = 1e-8 * var_scale * I_q
                Sigma_eps_y = 0.5 * (Sigma_eps_y + Sigma_eps_y.t()) + jitter_eps


                # Invert Sigma_eps_y stably via Cholesky
                try:
                    L_eps = torch.linalg.cholesky(Sigma_eps_y)
                    Y = torch.linalg.solve(L_eps, A_y)
                    M = I_q + eff_sigma_do_sq * (Y @ Y.t())
                    ei = 0.5 * torch.logdet(M)
                except Exception:
                    ei = torch.tensor(-1e9, dtype=torch.float64, device=device)

                if torch.isnan(ei) or torch.isinf(ei):
                    break

                # Backprop Euclidean gradient
                if W.grad is not None:
                    W.grad.zero_()
                ei.backward()

                with torch.no_grad():
                    grad_E = W.grad
                    if grad_E is None or torch.isnan(grad_E).any():
                        break

                    # Riemannian gradient on Stiefel: grad_R = grad_E - W @ grad_E.t() @ W
                    grad_R = grad_E - W @ (grad_E.t() @ W)

                    # Update with QR retraction
                    W_cand = W + current_lr * grad_R
                    Q, R = torch.linalg.qr(W_cand.t())
                    d = torch.diagonal(R)
                    ph = d / torch.abs(d)
                    W_new = (Q * ph).t()

                    W.copy_(W_new)

            # Evaluate final EI for this restart
            with torch.no_grad():
                if Sigma_x is not None:
                    Sigma_x_y = W @ Sigma_x @ W.t()
                    var_scale = torch.trace(Sigma_x_y) / float(q)
                    jitter_x = 1e-8 * var_scale * I_q
                    Sigma_x_y = 0.5 * (Sigma_x_y + Sigma_x_y.t()) + jitter_x
                    L_x = torch.linalg.cholesky(Sigma_x_y)
                    WASigmaW = W @ A @ Sigma_x @ W.t()
                    A_y = torch.linalg.solve(L_x.t(), torch.linalg.solve(L_x, WASigmaW.t()).t())
                else:
                    A_y = W @ A @ W.t()
                    var_scale = torch.trace(W @ Sigma_eps @ W.t()) / float(q)

                var_scale = torch.clamp(var_scale, min=1e-12)
                eff_sigma_do_sq = (kappa_do ** 2) * var_scale

                Sigma_eps_y = W @ Sigma_eps @ W.t()
                jitter_eps = 1e-8 * var_scale * I_q
                Sigma_eps_y = 0.5 * (Sigma_eps_y + Sigma_eps_y.t()) + jitter_eps

                try:
                    L_eps = torch.linalg.cholesky(Sigma_eps_y)
                    Y = torch.linalg.solve(L_eps, A_y)
                    M = I_q + eff_sigma_do_sq * (Y @ Y.t())
                    final_ei = float(0.5 * torch.logdet(M).item())
                except Exception:
                    final_ei = -float("inf")

                if final_ei > best_ei:
                    best_ei = final_ei
                    best_W = W.cpu().numpy().copy()


        return best_W, best_ei

except ImportError:
    optimize_stiefel_torch = None


def initialize_stiefel_matrix(p: int, q: int, method: str = "pca", cov_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Initializes orthogonal projection matrix W of shape (q, p) such that W @ W.T = I_q.
    """
    if method == "pca" and cov_matrix is not None:
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        top_indices = np.argsort(eigvals)[::-1][:q]
        W = eigvecs[:, top_indices].T
    else:
        Z = np.random.randn(p, q)
        Q, _ = np.linalg.qr(Z)
        W = Q.T
    return W


def optimize_coarse_graining_stiefel(
    A: np.ndarray,
    Sigma_eps: np.ndarray,
    q: int,
    Sigma_x: Optional[np.ndarray] = None,
    kappa_do: float = 1.0,
    n_restarts: int = 12,
    max_iter: int = 100,
    lr: float = 0.05
) -> Tuple[np.ndarray, float]:
    """
    Maximizes macro Effective Information over the Stiefel manifold V_q(R^p).
    """
    if optimize_stiefel_torch is not None:
        return optimize_stiefel_torch(
            A, Sigma_eps, q=q, Sigma_x_np=Sigma_x, kappa_do=kappa_do,
            n_restarts=n_restarts, max_iter=max_iter, lr=lr
        )

    # Fallback to PCA initialization if PyTorch is not available
    cov_for_pca = Sigma_x if Sigma_x is not None else Sigma_eps
    W_pca = initialize_stiefel_matrix(A.shape[0], q, method="pca", cov_matrix=cov_for_pca)
    from .analytical_ei import compute_macro_ei
    ei_pca = compute_macro_ei(A, Sigma_eps, W_pca, Sigma_x=Sigma_x, kappa_do=kappa_do)
    return W_pca, ei_pca
