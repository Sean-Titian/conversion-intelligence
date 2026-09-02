"""Authored synthetic event fixture for the point-in-time feature pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticEventConfig:
    sessions: int = 500
    users: int = 250
    seed: int = 20260901


@dataclass(frozen=True)
class SyntheticEventBundle:
    sessions: pd.DataFrame
    score_requests: pd.DataFrame
    page_events: pd.DataFrame


def _validate_config(config: SyntheticEventConfig) -> None:
    if config.sessions < 1:
        raise ValueError("sessions must be positive")
    if config.users < 1 or config.users > config.sessions:
        raise ValueError("users must be between one and the session count")


def generate_synthetic_event_bundle(
    config: SyntheticEventConfig | None = None,
) -> SyntheticEventBundle:
    """Generate generic events with boundary, future, and late-arrival cases.

    Every session has three requests (session start, 120 seconds, 300 seconds)
    and six page events. One event occurs exactly at the early request, another
    becomes available exactly at that request, one occurred before the late
    request but is not available until afterwards, and one occurs after every
    request. These cases make both cutoff boundaries and leakage gates observable.
    """

    if config is None:
        config = SyntheticEventConfig()
    _validate_config(config)
    rng = np.random.default_rng(config.seed)
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    countries = np.array(["US", "UK", "DE", "CA"])
    sources = np.array(["Ads", "Direct", "Seo"])
    categories = ("landing", "product", "comparison", "pricing", "checkout", "help")

    session_rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for index in range(config.sessions):
        session_id = f"synthetic_session_{index:08d}"
        user_id = f"synthetic_user_{index % config.users:08d}"
        session_start = base + pd.Timedelta(minutes=10 * index)
        session_rows.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "session_start_at": session_start,
                "country": str(rng.choice(countries)),
                "age": int(rng.integers(18, 76)),
                "new_user": int(rng.integers(0, 2)),
                "source": str(rng.choice(sources)),
            }
        )
        for suffix, offset_seconds in (("start", 0), ("early", 120), ("late", 300)):
            request_rows.append(
                {
                    "score_id": f"synthetic_score_{index:08d}_{suffix}",
                    "session_id": session_id,
                    "score_at": session_start + pd.Timedelta(seconds=offset_seconds),
                }
            )

        event_spec = (
            (30, 31),
            (90, 120),
            (120, 120),
            (180, 360),
            (240, 240),
            (360, 360),
        )
        for event_index, ((event_offset, available_offset), category) in enumerate(
            zip(event_spec, categories, strict=True)
        ):
            event_rows.append(
                {
                    "event_id": f"synthetic_event_{index:08d}_{event_index:02d}",
                    "session_id": session_id,
                    "event_at": session_start + pd.Timedelta(seconds=event_offset),
                    "available_at": session_start
                    + pd.Timedelta(seconds=available_offset),
                    "page_category": category,
                }
            )

    return SyntheticEventBundle(
        sessions=pd.DataFrame(session_rows),
        score_requests=pd.DataFrame(request_rows),
        page_events=pd.DataFrame(event_rows),
    )
