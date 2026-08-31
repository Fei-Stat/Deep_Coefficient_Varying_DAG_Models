# sanity_check.py
"""
Small implementation checks. These are unit tests, not a synthetic experiment.
"""

import torch

from model import DynamicDAG
from evaluate import extract_dag_adjacency, extract_raw_adjacency, is_dag


def test_nodewise_orientation():
    # beta[k, i, j] means i -> j.
    Y = torch.tensor([[2.0, 3.0, 5.0]])
    beta = torch.zeros(1, 3, 3)

    beta[0, 0, 1] = 4.0   # 0 -> 1
    beta[0, 1, 2] = -2.0  # 1 -> 2

    model = DynamicDAG(
        input_dim=1,
        hidden_dim=2,
        context_dim=1,
        n_nodes=3,
    )

    pred = model.nodewise_prediction(Y, beta)

    expected = torch.tensor([
        [0.0, 2.0 * 4.0, 3.0 * -2.0]
    ])

    assert torch.allclose(pred, expected), (pred, expected)


def test_spectral_chain_vs_cycle():
    model = DynamicDAG(
        input_dim=1,
        hidden_dim=2,
        context_dim=1,
        n_nodes=3,
        power_iteration_steps=50,
    )

    with torch.no_grad():
        model.alpha.zero_()

        # Chain 0 -> 1 -> 2
        model.alpha[0, 1, 0] = 1.0
        model.alpha[1, 2, 0] = 1.0

    rho_chain = float(model.exact_spectral_radius())
    assert abs(rho_chain) < 1e-7, rho_chain

    with torch.no_grad():
        # Add 2 -> 0 to create a directed 3-cycle.
        model.alpha[2, 0, 0] = 1.0

    rho_cycle = float(model.exact_spectral_radius())
    assert rho_cycle > 0.9, rho_cycle


def test_cycle_safe_extraction():
    S = torch.tensor(
        [
            [0.0, 0.9, 0.0],
            [0.0, 0.0, 0.8],
            [0.7, 0.0, 0.0],
        ]
    )

    raw = extract_raw_adjacency(S, threshold=0.1)
    assert not is_dag(raw)

    dag = extract_dag_adjacency(S, threshold=0.1)
    assert is_dag(dag)
    assert int(dag.sum()) == 2


if __name__ == "__main__":
    test_nodewise_orientation()
    test_spectral_chain_vs_cycle()
    test_cycle_safe_extraction()
    print("All sanity checks passed.")
