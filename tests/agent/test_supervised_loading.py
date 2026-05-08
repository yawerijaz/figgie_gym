from figgie_gym.agent.supervised import (
    SimplifiedExpectedValueGeometricAggressive,
    SupervisedModelAgent,
)


def test_load_checkpoint() -> None:
    checkpoint_path = "/Users/yawerijaz/mlflow_dir/mlartifacts/2/dbc0bfcb661c4cb2a9e5f5fd3748f05e/artifacts/epoch=3-step=6408/epoch=3-step=6408.ckpt"

    ev_calc = SimplifiedExpectedValueGeometricAggressive(
        full_pot=400,
        goal_symbol_value=10,
        num_symbols=4,
        late_aggressiveness_factor=1.1,
    )

    agent = SupervisedModelAgent.from_checkpoint(
        checkpoint_path=checkpoint_path,
        ev_calculator=ev_calc,
        quote_spread=2,
        apply_soft_clip=True,
        preprocessor="nested",
    )
    assert agent.model is not None


if __name__ == "__main__":
    test_load_checkpoint()
