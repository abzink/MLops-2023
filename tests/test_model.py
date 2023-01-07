import pytest
import torch

from src.models.model import MyAwesomeModel


class TestModel:
    # mark.parametrize enables parametrization of arguments for a test function
    # checks if certain input has the correct shape
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (torch.randn(1, 1, 2, 28), ValueError),
            (torch.randn(64, 1, 28, 2), ValueError),
            (torch.randn(100, 3, 28, 28), ValueError),
        ],
    )
    def test_model_input_shape(self, test_input, expected):
        model = MyAwesomeModel(0.25, 0.5)
        model.train()
        # if the input has the wrong shape, the model should raise a ValueError
        with pytest.raises(
            expected, match="Expected each x sample to have shape 1,28,28"
        ):
            model.forward(test_input)

    # checks if the output shape is correct
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (torch.randn(1, 1, 28, 28), (1, 10)),
            (torch.randn(64, 1, 28, 28), (64, 10)),
            (torch.randn(100, 1, 28, 28), (100, 10)),
        ],
    )
    def test_model_output(self, test_input, expected):
        model = MyAwesomeModel(0.25, 0.5)
        output = model(test_input)
        assert output.shape == expected
