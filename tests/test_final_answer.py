


import numpy as np
import PIL.Image
import pytest

from smolagents.agent_types import _AGENT_TYPE_MAPPING
from smolagents.default_tools import FinalAnswerTool

from .test_tools import ToolTesterMixin
from .utils.markers import require_torch


class TestFinalAnswerTool(ToolTesterMixin):
    def setup_method(self):
        self.inputs = {"answer": "Final answer"}
        self.tool = FinalAnswerTool()

    def test_exact_match_arg(self):
        result = self.tool("Final answer")
        assert result == "Final answer"

    def test_exact_match_kwarg(self):
        result = self.tool(answer=self.inputs["answer"])
        assert result == "Final answer"

    @require_torch
    def test_agent_type_output(self, inputs):
        for input_type, input in inputs.items():
            output = self.tool(**input, sanitize_inputs_outputs=True)
            agent_type = _AGENT_TYPE_MAPPING[input_type]
            assert isinstance(output, agent_type)

    @pytest.fixture
    def inputs(self, shared_datadir):
        import torch

        return {
            "string": {"answer": "Text input"},
            "image": {"answer": PIL.Image.open(shared_datadir / "000000039769.png").resize((512, 512))},
            "audio": {"answer": torch.Tensor(np.ones(3000))},
        }
