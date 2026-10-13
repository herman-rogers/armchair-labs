"""Application access to Opening Drive and its preserved comparison campaign."""

from engine.chairstats.store import CampaignNotFound, CampaignStore, safe_id

# Historical consolidation targets v2; dashboard selection is independent.
LATEST = "opening_drive_v2"
DEFAULT_CAMPAIGN = "opening_drive_v3"
APPLICATION_CAMPAIGNS = (DEFAULT_CAMPAIGN, LATEST)


class OpeningDriveStore(CampaignStore):
    """Keep historical research bundles outside application queries."""

    def campaign(self, campaign_id: str) -> dict:
        if safe_id(campaign_id) not in APPLICATION_CAMPAIGNS:
            raise CampaignNotFound(f"Unknown application campaign: {campaign_id}")
        return super().campaign(campaign_id)

    def campaigns(self) -> list[dict]:
        return [
            self.campaign(name) for name in APPLICATION_CAMPAIGNS if (self.root / name).is_dir()
        ]
