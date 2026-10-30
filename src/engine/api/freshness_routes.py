"""Health remains available when the analytical catalog cannot be queried."""

from fastapi import APIRouter

from engine.chairstats.application import DEFAULT_CAMPAIGN
from engine.config.settings import get_settings
from engine.data.freshness import freshness

router = APIRouter(prefix="/api/freshness", tags=["freshness"])


@router.get("")
def read_freshness(campaign: str = DEFAULT_CAMPAIGN):
    return freshness(get_settings().data_dir, campaign=campaign)
