"""Runtime helpers for CRT gripper eval/debug scripts.

This module intentionally avoids importing Isaac/omni UI modules so it can be
imported after AppLauncher by eval scripts without creating extra dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CRTGripperEvalGate:
    """Small state holder for manual eval policy gating.

    Start Pick in eval means "start policy output". Before that, the caller
    should hold the arm still and keep gripper auto-close disabled.
    """

    auto_start_policy: bool = False
    policy_started: bool = field(init=False)
    requests: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.requests.setdefault("start_pick_requested", False)
        self.requests.setdefault("reset_env_requested", False)
        self.policy_started = bool(self.auto_start_policy)

    def reset_after_env_reset(self) -> None:
        self.policy_started = bool(self.auto_start_policy)
        self.clear_requests()

    def clear_requests(self) -> None:
        self.requests["start_pick_requested"] = False
        self.requests["reset_env_requested"] = False

    def request_start_policy(self) -> None:
        self.requests["start_pick_requested"] = True

    def request_reset_env(self) -> None:
        self.requests["reset_env_requested"] = True

    def consume_start_policy(self) -> bool:
        return bool(self.requests.pop("start_pick_requested", False))

    def consume_reset_env(self) -> bool:
        return bool(self.requests.pop("reset_env_requested", False))

    def start_policy(self) -> None:
        self.policy_started = True
        self.requests["start_pick_requested"] = False


def get_arm_action_term(base_env: Any, action_name: str = "arm_action", require_processed_actions: bool = True) -> Any | None:
    """Return the IsaacLab action term used by the CRT reach policy."""

    manager = getattr(base_env, "action_manager", None)
    if manager is None:
        return None

    def usable(term: Any) -> bool:
        return term is not None and (not require_processed_actions or hasattr(term, "processed_actions"))

    if hasattr(manager, "get_term"):
        try:
            term = manager.get_term(action_name)
            if usable(term):
                return term
        except Exception:
            pass

    for attr in ("_terms", "_action_terms"):
        terms = getattr(manager, attr, None)
        if isinstance(terms, dict) and action_name in terms and usable(terms[action_name]):
            return terms[action_name]

    terms = getattr(manager, "_terms", None)
    if isinstance(terms, (list, tuple)):
        for term in terms:
            if usable(term):
                return term

    return None


def get_gripper_controller(base_env: Any, action_name: str = "arm_action") -> Any | None:
    """Return the gripper controller attached to the action term, if present."""

    term = get_arm_action_term(base_env, action_name=action_name, require_processed_actions=False)
    if term is None:
        return None
    return getattr(term, "gripper_controller", term)


def set_gripper_auto_close_enabled(base_env: Any, enabled: bool, action_name: str = "arm_action") -> bool:
    """Enable/disable deterministic gripper auto-close on the controller."""

    controller = get_gripper_controller(base_env, action_name=action_name)
    if controller is not None and hasattr(controller, "set_auto_close_enabled"):
        controller.set_auto_close_enabled(bool(enabled))
        return True
    return False


def request_gripper_close(base_env: Any, action_name: str = "arm_action") -> bool:
    """Force a gripper close request; mainly useful for low-level debug scripts."""

    controller = get_gripper_controller(base_env, action_name=action_name)
    if controller is not None and hasattr(controller, "request_close"):
        controller.request_close()
        return True
    return False
