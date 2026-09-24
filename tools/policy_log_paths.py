#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for grouping play/deploy logs by policy training run."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


POLICY_TIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def fallback_policy_time(prefix: str = "unknown_policy") -> str:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{stamp}"


def sanitize_path_token(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    value = value.strip(".-")
    return value or fallback_policy_time()


def extract_policy_time_from_text(text: str | None) -> str | None:
    if not text:
        return None
    match = POLICY_TIME_RE.search(str(text))
    return match.group(0) if match else None


def infer_policy_time_from_checkpoint(checkpoint: str | Path | None, fallback: str | None = None) -> str:
    if checkpoint:
        path = Path(checkpoint).expanduser()
        candidates = [
            path.parent.name,
            path.parent.parent.name,
            path.stem,
            str(path),
        ]
        for text in candidates:
            policy_time = extract_policy_time_from_text(text)
            if policy_time:
                return policy_time
        for text in candidates:
            if text:
                return sanitize_path_token(text)
    return fallback or fallback_policy_time()


def infer_policy_time_from_csv(csv_path: str | Path | None, log_name: str) -> str | None:
    if not csv_path:
        return None

    path = Path(csv_path).expanduser()
    grouped_prefix = f"{log_name}-"
    parent_name = path.parent.name
    if parent_name.startswith(grouped_prefix):
        policy_time = extract_policy_time_from_text(parent_name[len(grouped_prefix):])
        if policy_time:
            return policy_time

    return extract_policy_time_from_text(path.stem) or extract_policy_time_from_text(parent_name)


def policy_log_dir(root: str | Path, log_name: str, policy_time: str) -> Path:
    return Path(root).expanduser() / f"{log_name}-{sanitize_path_token(policy_time)}"


def avoid_overwrite(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_run{index:02d}{path.suffix}")
        if not candidate.exists():
            return candidate

    return path.with_name(f"{path.stem}_run{fallback_policy_time('extra')}{path.suffix}")


def make_policy_output_path(
    root: str | Path,
    log_name: str,
    policy_time: str,
    file_prefix: str,
    suffix: str = ".csv",
    dedupe: bool = True,
) -> Path:
    out_dir = policy_log_dir(root, log_name, policy_time)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sanitize_path_token(file_prefix)}_{sanitize_path_token(policy_time)}{suffix}"
    return avoid_overwrite(path) if dedupe else path
