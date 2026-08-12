from pathlib import Path
import unittest


class StreamlitDeploymentTests(unittest.TestCase):
    def test_entrypoint_and_dependencies_exist(self):
        self.assertTrue(Path("app/Home.py").is_file())
        requirements = Path("requirements.txt").read_text()
        self.assertIn("streamlit", requirements)
        self.assertIn("psycopg2-binary", requirements)
        self.assertIn("langgraph-checkpoint-postgres", requirements)

    def test_secrets_are_not_tracked(self):
        self.assertFalse(Path(".env").exists())
        self.assertFalse(Path(".streamlit/secrets.toml").exists())


if __name__ == "__main__":
    unittest.main()
