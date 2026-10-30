"""Game engine replay data: tracked games, their plays, and per-play trajectories."""

from fastapi import APIRouter, HTTPException

from engine.config.settings import get_settings
from engine.data import tracking_viewer

router = APIRouter(prefix="/api/tracking", tags=["Tracking replay"])


def _check_id(value: str, label: str) -> str:
    if not value.isdigit() or len(value) > 12:
        raise HTTPException(404, f"Unknown {label}: {value}")
    return value


def _source(source_id: str) -> str:
    if source_id not in tracking_viewer.SOURCES:
        raise HTTPException(404, f"Unknown tracking source: {source_id}")
    return source_id


@router.get("/sources")
def sources():
    """Every replayable tracking archive, with availability and game counts."""
    return {"sources": tracking_viewer.sources(get_settings().data_dir)}


@router.get("/{source_id}/games")
def games(source_id: str):
    try:
        return {"games": tracking_viewer.games(get_settings().data_dir, _source(source_id))}
    except (OSError, ValueError, TypeError, StopIteration) as exc:
        raise HTTPException(409, f"Tracking source unavailable: {exc}") from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0] if exc.args else exc)) from exc


@router.get("/{source_id}/games/{game_id}/plays")
def plays(source_id: str, game_id: str):
    try:
        return {
            "plays": tracking_viewer.plays(
                get_settings().data_dir, _source(source_id), _check_id(game_id, "game")
            )
        }
    except (OSError, ValueError, TypeError, StopIteration) as exc:
        raise HTTPException(409, f"Tracking source unavailable: {exc}") from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0] if exc.args else exc)) from exc


@router.get("/{source_id}/games/{game_id}/plays/{play_id}")
def play(source_id: str, game_id: str, play_id: str):
    try:
        return tracking_viewer.play(
            get_settings().data_dir,
            _source(source_id),
            _check_id(game_id, "game"),
            _check_id(play_id, "play"),
        )
    except (OSError, ValueError, TypeError, StopIteration) as exc:
        raise HTTPException(409, f"Tracking source unavailable: {exc}") from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0] if exc.args else exc)) from exc
