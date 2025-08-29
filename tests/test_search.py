


from smolagents import DuckDuckGoSearchTool

from .test_tools import ToolTesterMixin
from .utils.markers import require_run_all


class TestDuckDuckGoSearchTool(ToolTesterMixin):
    def setup_method(self):
        self.tool = DuckDuckGoSearchTool()
        self.tool.setup()

    @require_run_all
    def test_exact_match_arg(self):
        result = self.tool("Agents")
        assert isinstance(result, str)

    @require_run_all
    def test_agent_type_output(self):
        super().test_agent_type_output()
