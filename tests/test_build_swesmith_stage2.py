from scripts.build_swesmith_stage2_key_actions import (
    classify_candidates,
    is_source_path,
    parse_action,
)


def action(tool: str, **params: str) -> str:
    body = "\n".join(f"<parameter={key}>{value}</parameter>" for key, value in params.items())
    return f"thinking\n<function={tool}>\n{body}\n</function>"


def test_parse_action_extracts_xml_parameters():
    parsed = parse_action(action("str_replace_editor", command="view", path="/testbed/pkg/mod.py"))
    assert parsed is not None
    assert parsed.tool == "str_replace_editor"
    assert parsed.params == {"command": "view", "path": "/testbed/pkg/mod.py"}


def test_source_path_excludes_tests_and_reproduction_scripts():
    assert is_source_path("/testbed/pkg/mod.py")
    assert not is_source_path("/testbed/tests/test_mod.py")
    assert not is_source_path("/testbed/reproduce_bug.py")


def test_classification_covers_key_stage2_actions():
    conversations = [
        {"from": "human", "value": "fix it"},
        {"from": "gpt", "value": action("bash", command="grep -R symbol /testbed/pkg")},
        {"from": "human", "value": "/testbed/pkg/mod.py:10:symbol"},
        {"from": "gpt", "value": action("str_replace_editor", command="view", path="/testbed/pkg/mod.py")},
        {"from": "human", "value": "file contents"},
        {
            "from": "gpt",
            "value": action(
                "str_replace_editor",
                command="str_replace",
                path="/testbed/pkg/mod.py",
                old_str="bad",
                new_str="good",
            ),
        },
        {"from": "human", "value": "edit applied"},
        {"from": "gpt", "value": action("bash", command="python -m pytest tests/test_mod.py")},
        {"from": "human", "value": "1 passed"},
        {"from": "gpt", "value": action("submit")},
    ]
    candidates, _ = classify_candidates(conversations, 3)
    assert [candidate.category for candidate in candidates] == [
        "localization",
        "localization",
        "source_edit",
        "validation",
        "submit",
    ]


def test_exact_repeat_after_editor_failure_is_not_positive_recovery():
    failed_action = action(
        "str_replace_editor",
        command="str_replace",
        path="/testbed/pkg/mod.py",
        old_str="missing",
        new_str="fixed",
    )
    conversations = [
        {"from": "human", "value": "fix it"},
        {"from": "gpt", "value": failed_action},
        {"from": "human", "value": "No replacement was performed: old_str did not appear verbatim"},
        {"from": "gpt", "value": failed_action},
    ]
    candidates, rejected = classify_candidates(conversations, 1)
    assert [candidate.category for candidate in candidates] == ["source_edit"]
    assert rejected == 1


def test_test_failure_followed_by_source_edit_is_recovery():
    conversations = [
        {"from": "human", "value": "fix it"},
        {"from": "gpt", "value": action("bash", command="python -m pytest tests/test_mod.py")},
        {"from": "human", "value": "FAILED tests/test_mod.py::test_case - AssertionError"},
        {
            "from": "gpt",
            "value": action(
                "str_replace_editor",
                command="str_replace",
                path="/testbed/pkg/mod.py",
                old_str="bad",
                new_str="good",
            ),
        },
    ]
    candidates, _ = classify_candidates(conversations, 2)
    assert candidates[-1].category == "recovery"
    assert candidates[-1].subtype == "test_failure"


def test_cleanup_and_submit_are_not_recovery_targets():
    conversations = [
        {"from": "human", "value": "fix it"},
        {
            "from": "gpt",
            "value": action(
                "str_replace_editor",
                command="str_replace",
                path="/testbed/pkg/mod.py",
                old_str="bad",
                new_str="good",
            ),
        },
        {"from": "human", "value": "No replacement was performed: old_str did not appear verbatim"},
        {"from": "gpt", "value": action("bash", command="rm /testbed/reproduce.py")},
        {"from": "human", "value": "Traceback (most recent call last): test failed"},
        {"from": "gpt", "value": action("submit")},
    ]
    candidates, _ = classify_candidates(conversations, 4)
    assert all(candidate.category != "recovery" for candidate in candidates)
