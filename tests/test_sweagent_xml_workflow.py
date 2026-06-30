from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from agent.sweagent_xml_workflow import (
    SWEAgentXMLConfig,
    SWEAgentXMLWorkflow,
    StrReplaceEditor,
    parse_xml_function_action,
)
from tools.executor import ToolExecutor


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def generate(self, messages: list[dict[str, str]], **overrides: object) -> str:
        return next(self.responses)


class TokenCountingLLM(FakeLLM):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.seen_messages: list[list[dict[str, str]]] = []

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages) // 4

    def generate(self, messages: list[dict[str, str]], **overrides: object) -> str:
        self.seen_messages.append([dict(message) for message in messages])
        return super().generate(messages, **overrides)


def action(name: str, **params: object) -> str:
    body = "\n".join(f"<parameter={key}>{value}</parameter>" for key, value in params.items())
    return f"Reasoning before the tool call.\n<function={name}>\n{body}\n</function>"


class XMLParserTests(unittest.TestCase):
    def test_parses_all_supported_tools(self) -> None:
        bash = parse_xml_function_action(action("bash", command="find /testbed -name '*.py'"))
        editor = parse_xml_function_action(
            action("str_replace_editor", command="view", path="/testbed/a.py", view_range="[2, -1]")
        )
        submit = parse_xml_function_action(action("submit"))

        self.assertEqual(bash.params["command"], "find /testbed -name '*.py'")
        self.assertEqual(editor.params["view_range"], [2, -1])
        self.assertEqual(submit.params, {})

    def test_rejects_multiple_or_unknown_calls(self) -> None:
        with self.assertRaises(ValueError):
            parse_xml_function_action(action("bash", command="ls") + action("submit"))
        with self.assertRaises(ValueError):
            parse_xml_function_action(action("other"))


class EditorTests(unittest.TestCase):
    def test_replace_insert_and_undo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "module.py"
            path.write_text("one\ntwo\n", encoding="utf-8")
            editor = StrReplaceEditor(root)

            replaced = editor.execute(
                {"command": "str_replace", "path": "/testbed/module.py", "old_str": "two", "new_str": "three"}
            )
            inserted = editor.execute(
                {"command": "insert", "path": "/testbed/module.py", "insert_line": 1, "new_str": "middle"}
            )
            undone = editor.execute({"command": "undo_edit", "path": "/testbed/module.py"})

            self.assertTrue(replaced.ok)
            self.assertTrue(inserted.ok)
            self.assertTrue(undone.ok)
            self.assertEqual(path.read_text(encoding="utf-8"), "one\nthree\n")

    def test_long_python_view_uses_official_filemap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "\n".join(f"    value += {i}  # padding padding padding" for i in range(600))
            (root / "large.py").write_text(
                f"def long_function(value):\n{body}\n    return value\n\ndef other():\n    return 1\n",
                encoding="utf-8",
            )

            result = StrReplaceEditor(root).execute({"command": "view", "path": "/testbed/large.py"})

            self.assertTrue(result.ok)
            self.assertIn("This file is too large to display entirely", result.output)
            self.assertIn("eliding lines", result.output)
            self.assertIn("def other():", result.output)


class WorkflowTests(unittest.TestCase):
    def test_shell_preserves_working_directory_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "package").mkdir()
            tools = ToolExecutor(root)

            changed = tools.run_shell("cd package && export SWE_TEST_VALUE=preserved")
            observed = tools.run_shell('printf "%s|%s" "$PWD" "$SWE_TEST_VALUE"')

            self.assertTrue(changed.ok)
            self.assertTrue(observed.ok)
            self.assertIn(f"{root}/package|preserved", observed.output)

    def test_long_shell_output_keeps_head_and_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolExecutor(tmp, max_output_chars=160)

            result = tools.run_shell("python -c \"print('HEAD' + 'x' * 400 + 'TAIL')\"")

            self.assertIn("HEAD", result.output)
            self.assertIn("TAIL", result.output)
            self.assertIn("middle omitted", result.output)

    def test_failed_shell_action_has_structured_recovery_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workflow = SWEAgentXMLWorkflow(
                llm=FakeLLM([action("bash", command="false")]),
                tools=ToolExecutor(tmp),
                config=SWEAgentXMLConfig([], [], max_steps=1),
            )

            result = workflow.run("Run a failing command")

            self.assertEqual(result.steps[0].result.error_type, "command_failed")
            self.assertIn("non-zero status", result.messages[-1]["content"])

    def test_stops_after_two_empty_submissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            workflow = SWEAgentXMLWorkflow(
                llm=FakeLLM([action("submit"), action("submit")]),
                tools=ToolExecutor(root),
                config=SWEAgentXMLConfig([], [], max_steps=10),
            )

            result = workflow.run("Submit without a patch")

            self.assertFalse(result.finished)
            self.assertEqual([step.tool for step in result.steps], ["submit", "submit", "empty_submit_limit"])
            self.assertTrue(all(step.result.error_type == "empty_submit" for step in result.steps[:2]))

    def test_blocks_and_stops_identical_failed_editor_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            invalid = action("str_replace_editor", command="view", path="/testbed/missing.py")
            workflow = SWEAgentXMLWorkflow(
                llm=FakeLLM([invalid] * 5),
                tools=ToolExecutor(root),
                config=SWEAgentXMLConfig([], [], max_steps=10),
            )

            result = workflow.run("Read a missing file repeatedly")

            self.assertFalse(result.finished)
            self.assertEqual(result.steps[-1].tool, "repeated_action_limit")
            self.assertEqual(
                [step.result.error_type for step in result.steps[:-1]],
                ["editor_error", "editor_error", "repeated_action", "repeated_action", "repeated_action"],
            )

    def test_editor_review_and_second_submit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "module.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-qm", "initial"],
                cwd=root,
                check=True,
            )
            llm = FakeLLM(
                [
                    action(
                        "str_replace_editor",
                        command="str_replace",
                        path="/testbed/module.py",
                        old_str="VALUE = 1",
                        new_str="VALUE = 2",
                    ),
                    action("submit"),
                    action("submit"),
                ]
            )
            workflow = SWEAgentXMLWorkflow(
                llm=llm,
                tools=ToolExecutor(root, max_output_chars=16000),
                config=SWEAgentXMLConfig([], [], max_steps=3),
            )

            result = workflow.run("Change VALUE to 2")

            self.assertTrue(result.finished)
            self.assertEqual([step.tool for step in result.steps], ["str_replace_editor", "submit", "submit"])
            self.assertTrue(
                any(
                    message["role"] == "user" and "Thank you for your work" in message["content"]
                    for message in result.messages
                )
            )
            self.assertIn("+VALUE = 2", result.summary)
            status = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True)
            self.assertEqual(status.stdout, " M module.py\n")

    def test_bash_maps_virtual_testbed_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "marker.txt").write_text("ok\n", encoding="utf-8")
            workflow = SWEAgentXMLWorkflow(
                llm=FakeLLM([action("bash", command="cat /testbed/marker.txt")]),
                tools=ToolExecutor(root),
                config=SWEAgentXMLConfig([], [], max_steps=1),
            )

            result = workflow.run("Read marker")

            self.assertEqual(result.steps[0].result.output, "ok")

    def test_autosubmits_existing_patch_when_context_limit_is_reached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "module.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-qm", "initial"],
                cwd=root,
                check=True,
            )
            llm = TokenCountingLLM(
                [
                    "x" * 4_000
                    + action(
                        "str_replace_editor",
                        command="str_replace",
                        path="/testbed/module.py",
                        old_str="VALUE = 1",
                        new_str="VALUE = 2",
                    ),
                ]
            )
            workflow = SWEAgentXMLWorkflow(
                llm=llm,
                tools=ToolExecutor(root),
                config=SWEAgentXMLConfig(
                    [],
                    [],
                    max_steps=2,
                    hard_context_tokens=1_800,
                ),
            )

            result = workflow.run("Change VALUE to 2")

            self.assertTrue(result.finished)
            self.assertEqual(len(llm.seen_messages), 1)
            self.assertEqual(result.steps[-1].tool, "context_limit")
            self.assertIn("+VALUE = 2", result.summary)

    def test_stops_before_generation_when_initial_context_exceeds_hard_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            llm = TokenCountingLLM([])
            workflow = SWEAgentXMLWorkflow(
                llm=llm,
                tools=ToolExecutor(root),
                config=SWEAgentXMLConfig(
                    [],
                    [],
                    hard_context_tokens=5_000,
                ),
            )

            result = workflow.run("x" * 20_000)

            self.assertFalse(result.finished)
            self.assertEqual(llm.seen_messages, [])
            self.assertEqual(result.steps[-1].tool, "context_limit")
            self.assertIn("No valid patch", result.summary)

    def test_bash_output_hides_real_workspace_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = SWEAgentXMLWorkflow(
                llm=FakeLLM([action("bash", command="pwd")]),
                tools=ToolExecutor(root),
                config=SWEAgentXMLConfig([], [], max_steps=1),
            )

            result = workflow.run("Show the current directory")

            self.assertEqual(result.steps[0].result.output, "/testbed")
            self.assertNotIn(str(root), result.steps[0].result.output)

    def test_stops_after_three_consecutive_command_timeouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = SWEAgentXMLWorkflow(
                llm=FakeLLM([action("bash", command="sleep 1")] * 3),
                tools=ToolExecutor(root),
                config=SWEAgentXMLConfig(
                    [],
                    [],
                    max_steps=4,
                    test_timeout=0.01,
                    total_execution_timeout=0,
                    max_consecutive_command_timeouts=3,
                ),
            )

            result = workflow.run("Run a slow command")

            self.assertFalse(result.finished)
            self.assertEqual(result.steps[-1].tool, "command_timeout_limit")

    def test_autosubmits_after_total_execution_time_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = SWEAgentXMLWorkflow(
                llm=FakeLLM([action("bash", command="sleep 0.03")]),
                tools=ToolExecutor(root),
                config=SWEAgentXMLConfig(
                    [],
                    [],
                    max_steps=2,
                    total_execution_timeout=0.01,
                ),
            )

            result = workflow.run("Run a command")

            self.assertFalse(result.finished)
            self.assertEqual(result.steps[-1].tool, "execution_time_limit")


if __name__ == "__main__":
    unittest.main()
