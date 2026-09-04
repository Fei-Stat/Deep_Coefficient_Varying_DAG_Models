"""Fast implementation checks for the current end-to-end model."""

import torch

from models.TransMIL_regression import (
    DynamicDAG,
    TransMILDAG,
    TransMILRegression,
    extract_dag_adjacency,
    is_dag,
)


def test_nodewise_orientation_and_intercept():
    Y = torch.tensor([[2.0, 3.0, 5.0]])
    beta = torch.zeros(1, 3, 3)
    beta[0, 0, 1] = 4.0   # 0 -> 1
    beta[0, 1, 2] = -2.0  # 1 -> 2
    intercept = torch.tensor([[1.0, 1.0, 1.0]])

    prediction = DynamicDAG.nodewise_prediction(Y, beta, intercept)
    expected = torch.tensor([[1.0, 9.0, -5.0]])
    assert torch.allclose(prediction, expected), (prediction, expected)


def test_spectral_chain_vs_cycle():
    model = DynamicDAG(
        input_dim=2,
        hidden_dim=4,
        context_dim=1,
        n_nodes=3,
        power_iteration_steps=50,
    )
    with torch.no_grad():
        model.alpha.zero_()
        model.alpha[0, 1, 0] = 1.0
        model.alpha[1, 2, 0] = 1.0
    assert float(model.exact_spectral_radius()) < 1e-7

    with torch.no_grad():
        model.alpha[2, 0, 0] = 1.0
    assert float(model.exact_spectral_radius()) > 0.9


def test_cycle_safe_extraction():
    strength = torch.tensor(
        [[0.0, 0.9, 0.0], [0.0, 0.0, 0.8], [0.7, 0.0, 0.0]]
    )
    raw = (strength > 0.1).to(torch.int64)
    assert not is_dag(raw)
    safe = extract_dag_adjacency(strength, threshold=0.1)
    assert is_dag(safe)
    assert int(safe.sum()) == 2


def test_dag_bypasses_regression_head():
    dag = TransMILDAG(
        n_genes=4,
        feat_dim=32,
        dag_hidden_dim=8,
        context_dim=3,
    )
    baseline = TransMILRegression(n_genes=4, feat_dim=32)
    assert not hasattr(dag, "regression_head")
    assert hasattr(baseline, "regression_head")


def test_wsi_only_graph_inference():
    torch.manual_seed(0)
    model = TransMILDAG(
        n_genes=4,
        feat_dim=32,
        dag_hidden_dim=8,
        context_dim=3,
        global_edge_threshold=0.0,
        patient_edge_threshold=0.0,
    )
    model.eval()
    features = torch.randn(2, 16, 32)
    with torch.no_grad():
        output = model.infer_graph(features)

    assert output["embedding"].shape == (2, 512)
    assert output["beta"].shape == (2, 4, 4)
    assert output["intercept"].shape == (2, 4)
    assert output["raw_global_adjacency"].shape == (4, 4)
    assert output["patient_adjacency"].shape == (2, 4, 4)
    assert is_dag(output["global_adjacency"])
    assert all(is_dag(graph) for graph in output["patient_adjacency"])


def test_paired_sem_training_output():
    torch.manual_seed(1)
    model = TransMILDAG(
        n_genes=4,
        feat_dim=32,
        dag_hidden_dim=8,
        context_dim=3,
    )
    features = torch.randn(2, 16, 32)
    expression = torch.randn(2, 4)
    output = model(
        data=features,
        Y=expression,
        lambda_group=1e-3,
        gamma_acyclicity=1e-2,
    )
    assert output["Y_hat"].shape == expression.shape
    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.dynamic_dag.alpha.grad is not None


def run_all():
    tests = [
        test_nodewise_orientation_and_intercept,
        test_spectral_chain_vs_cycle,
        test_cycle_safe_extraction,
        test_dag_bypasses_regression_head,
        test_wsi_only_graph_inference,
        test_paired_sem_training_output,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} sanity checks passed.")


if __name__ == "__main__":
    run_all()
