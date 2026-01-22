from examples.backtest_v3_stress import (
    generate_parameter_grid,
    parse_float_grid,
    parse_int_grid,
    parse_str_grid,
)


def test_parse_helpers_handle_whitespace_and_numeric():
    assert parse_float_grid("0.1, 0.20 ,0.3") == [0.1, 0.2, 0.3]
    assert parse_int_grid("1,2, 3") == [1, 2, 3]
    assert parse_str_grid("2021-01-01 , 2022-01-01") == ["2021-01-01", "2022-01-01"]


def test_generate_parameter_grid_cartesian_product():
    grid_axes = {
        "slippage": [0.01, 0.02],
        "transaction_cost_rate": [0.01],
        "train_window_size": [9, 12],
    }
    combos = generate_parameter_grid(grid_axes)
    assert len(combos) == 4  # 2 * 1 * 2
    assert {"slippage": 0.01, "transaction_cost_rate": 0.01, "train_window_size": 9} in combos
    assert {"slippage": 0.02, "transaction_cost_rate": 0.01, "train_window_size": 12} in combos
