"""Test preparation of permanent README links for PyPI."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_pypi_readme as pypi_readme


class PreparePyPIReadmeTest(unittest.TestCase):
    repository = "owner/project"
    revision = "a" * 40

    def test_rewrites_relative_links_only(self):
        markdown = "\n".join(
            (
                "[guide](docs/guide.md#install)",
                "![diagram](assets/diagram.svg)",
                "[external](https://example.com/docs)",
                "[email](mailto:maintainer@example.com)",
                "[section](#install)",
            )
        )

        rewritten, count = pypi_readme.rewrite_links(markdown, self.repository, self.revision)

        base = f"https://github.com/{self.repository}/blob/{self.revision}"
        self.assertEqual(count, 2)
        self.assertIn(f"[guide]({base}/docs/guide.md#install)", rewritten)
        self.assertIn(f"![diagram]({base}/assets/diagram.svg)", rewritten)
        self.assertIn("[external](https://example.com/docs)", rewritten)
        self.assertIn("[email](mailto:maintainer@example.com)", rewritten)
        self.assertIn("[section](#install)", rewritten)

    def test_prepares_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            readme = Path(tmp) / "README.md"
            readme.write_text("See [license](LICENSE).\n")

            count = pypi_readme.prepare(readme, self.repository, self.revision)

            self.assertEqual(count, 1)
            self.assertEqual(
                readme.read_text(),
                f"See [license](https://github.com/{self.repository}/blob/"
                f"{self.revision}/LICENSE).\n",
            )

    def test_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "OWNER/NAME"):
            pypi_readme.rewrite_links("[guide](docs/guide.md)", "owner", self.revision)
        with self.assertRaisesRegex(ValueError, "full lowercase"):
            pypi_readme.rewrite_links("[guide](docs/guide.md)", self.repository, "main")
        with self.assertRaisesRegex(ValueError, "escapes"):
            pypi_readme.rewrite_links("[outside](../README.md)", self.repository, self.revision)


if __name__ == "__main__":
    unittest.main()
