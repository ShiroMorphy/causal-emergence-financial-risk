import numpy as np
import torch

from causal_emergence.cuda_stiefel import evaluate_batch_cefi


def _systems():
    A1 = np.array([[0.35, 0.12, 0.0], [-0.05, 0.2, 0.08], [0.02, 0.0, 0.15]])
    A2 = np.array([[0.2, -0.03, 0.04], [0.1, 0.25, 0.0], [0.0, 0.07, 0.3]])
    S1 = np.array([[0.8, 0.1, 0.0], [0.1, 1.1, 0.05], [0.0, 0.05, 0.9]])
    S2 = np.array([[1.0, -0.05, 0.02], [-0.05, 0.7, 0.0], [0.02, 0.0, 1.2]])
    X1 = np.diag([1.1, 0.9, 1.0])
    X2 = np.diag([0.8, 1.2, 1.1])
    return np.stack([A1, A2]), np.stack([S1, S2]), np.stack([X1, X2])


def test_batch_partition_invariance():
    A, S, X = _systems()
    kwargs = dict(q_candidates=[1, 2], n_restarts=3, max_iter=8, device=torch.device("cpu"))
    together = evaluate_batch_cefi(A, S, X, **kwargs)
    first = evaluate_batch_cefi(A[:1], S[:1], X[:1], **kwargs)
    second = evaluate_batch_cefi(A[1:], S[1:], X[1:], **kwargs)

    np.testing.assert_allclose(together[0], np.r_[first[0], second[0]], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(together[1], np.r_[first[1], second[1]])
    np.testing.assert_allclose(together[2], np.r_[first[2], second[2]], rtol=0, atol=1e-12)


def test_micro_ei_is_isotropic_and_in_nats():
    A, S, X = _systems()
    result = evaluate_batch_cefi(
        A[:1], S[:1], X[:1], [1, 2], n_restarts=2, max_iter=3, device=torch.device("cpu")
    )
    sigma_sq = np.trace(X[0]) / 3.0
    S_clean = 0.5 * (S[0] + S[0].T) + 1e-10 * sigma_sq * np.eye(3)
    expected = 0.5 * (
        np.linalg.slogdet(sigma_sq * A[0] @ A[0].T + S_clean)[1]
        - np.linalg.slogdet(S_clean)[1]
    )
    np.testing.assert_allclose(result[2][0], expected, rtol=0, atol=1e-12)


def test_best_projection_is_orthonormal():
    A, S, X = _systems()
    result = evaluate_batch_cefi(
        A[:1], S[:1], X[:1], [2], n_restarts=2, max_iter=3,
        device=torch.device("cpu"), return_best_w=True,
    )
    W = result[4][0]
    np.testing.assert_allclose(W @ W.T, np.eye(W.shape[0]), atol=1e-10)


def test_cuda_global_near_tie_rule():
    """Regression guard for chained near-ties across ascending q values."""
    # The analytical test above exercises the exact counterexample. Here we
    # assert the shared public rule directly to prevent the CUDA API from
    # drifting to ordinary/streaming argmax semantics.
    from causal_emergence.cuda_stiefel import _select_q_with_global_tolerance

    spectrum = torch.tensor([[0.0, 0.9e-7, 1.1e-7]], dtype=torch.float64)
    selected_idx, q_star = _select_q_with_global_tolerance(
        spectrum, [1, 2, 3], 1e-7
    )
    assert selected_idx.item() == 1
    assert q_star.item() == 2
