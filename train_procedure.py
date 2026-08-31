# ============================================================
# Complete training procedure
# ============================================================

def train_model(
    model,
    z,
    Y,
    observational_mask=None,
    config=None
):

    if config is None:
        config = TrainConfig()

    set_seed(
        config.seed
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = next(
        model.parameters()
    ).device

    z = z.to(device)
    Y = Y.to(device)

    if observational_mask is not None:

        observational_mask = (
            observational_mask
            .to(device)
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = build_optimizer(
        model,
        config
    )

    history = []

    # ========================================================
    # Stage I
    # ========================================================

    train_stage1(
        model=model,
        optimizer=optimizer,

        z=z,
        Y=Y,

        observational_mask=
            observational_mask,

        config=config,
        history=history
    )

    # ========================================================
    # Optional Stage-I edge screening
    # ========================================================

    edge_mask = None

    if config.use_screening:

        edge_mask = (
            build_screening_mask(
                model,
                config
            )
        )

        n_edges = int(
            edge_mask.sum().item()
        )

        print(
            f"\nCandidate edges after "
            f"screening: {n_edges}"
        )

        if n_edges == 0:

            raise RuntimeError(
                "Stage-I screening removed "
                "all candidate edges. "
                "Decrease screening_threshold."
            )

        enforce_parameter_mask(
            model,
            edge_mask=edge_mask
        )

    # ========================================================
    # Stage II
    # ========================================================

    train_stage2(
        model=model,
        optimizer=optimizer,

        z=z,
        Y=Y,

        observational_mask=
            observational_mask,

        config=config,
        history=history,

        edge_mask=edge_mask
    )

    # ========================================================
    # Final graph
    # ========================================================

    S, adjacency = extract_adjacency(
        model,
        threshold=
            config.edge_threshold
    )

    dag_flag = is_dag(
        adjacency
    )

    # ========================================================
    # Final beta
    # ========================================================

    model.eval()

    with torch.no_grad():

        x = model.encode_context(z)

        beta = model.compute_beta(x)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(
        "\n"
        "====================================\n"
        "Training complete\n"
        "===================================="
    )

    print(
        f"Final spectral radius: "
        f"{float(model.spectral_acyclicity()):.3e}"
    )

    print(
        f"Number of selected edges: "
        f"{int(adjacency.sum())}"
    )

    print(
        f"Thresholded graph is DAG: "
        f"{dag_flag}"
    )

    return {
        "model": model,
        "history": history,

        "support": S.detach(),
        "adjacency": adjacency.detach(),

        "beta": beta.detach(),
        "context": x.detach(),

        "is_dag": dag_flag,

        "screening_mask": (
            None
            if edge_mask is None
            else edge_mask.detach()
        )
    }
