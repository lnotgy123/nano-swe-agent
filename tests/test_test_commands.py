from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bench.test_commands import command_argv, test_command as build_test_command
from bench.evaluator import _timeout_output_text


class DjangoTestCommandTests(unittest.TestCase):
    def test_converts_django_unittest_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "runtests.py").touch()

            command = command_argv("test_method (test_app.tests.ExampleTests)", root)

            self.assertEqual(
                command,
                ["python", "tests/runtests.py", "--verbosity", "1", "test_app.tests.ExampleTests.test_method"],
            )

    def test_skips_non_executable_django_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "runtests.py").touch()
            description = "assertRaisesMessage shouldn't interpret RE special chars."

            self.assertIsNone(command_argv(description, root))
            self.assertEqual(
                build_test_command([description, "test_ok (test_app.tests.ExampleTests)"], root),
                "python tests/runtests.py --verbosity 1 test_app.tests.ExampleTests.test_ok",
            )

    def test_timeout_output_accepts_subprocess_bytes(self) -> None:
        self.assertEqual(_timeout_output_text(b"partial output\xff"), "partial output�")
        self.assertEqual(_timeout_output_text(None), "")


if __name__ == "__main__":
    unittest.main()
