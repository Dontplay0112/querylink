import copy
import sys
import tomllib
import unittest
from pathlib import Path

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from components.config import MyConfig  # noqa: E402


class ConfigTestCase(unittest.TestCase):
    def load_raw_config(self, name: str) -> dict:
        with (PROJECT_ROOT / "config" / name).open("rb") as config_file:
            return tomllib.load(config_file)

    def test_release_configs_validate(self):
        expected = {
            "locomo.toml": ("locomo", "data/locomo"),
            "longmemeval.toml": ("longmemeval", "data/longmemeval"),
        }

        for name, (task, data_path) in expected.items():
            with self.subTest(config=name):
                config = MyConfig.model_validate(self.load_raw_config(name))
                self.assertEqual(config.exp.task, task)
                self.assertEqual(str(config.exp.data_path), data_path)
                self.assertEqual(config.agent.llm_model_source, "openai")
                self.assertEqual(config.agent.llm_model_name, "gpt-4o-mini")

    def test_unknown_agent_option_is_rejected(self):
        raw_config = copy.deepcopy(self.load_raw_config("locomo.toml"))
        raw_config["agent"]["obsolete_option"] = True

        with self.assertRaises(ValidationError):
            MyConfig.model_validate(raw_config)


if __name__ == "__main__":
    unittest.main()
