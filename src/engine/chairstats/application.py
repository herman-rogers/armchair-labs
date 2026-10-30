"""Application access to Opening Drive: every bundle in the campaign directory.

The directory is the registry. Any folder holding a ``manifest.json`` is a model the
API serves; nothing is allowlisted. The default is a preference, not a gate: when it
is absent the newest campaign is used.
"""

import warnings

from engine.chairstats.store import CampaignStore

DEFAULT_CAMPAIGN = "opening_drive_v41"


class OpeningDriveStore(CampaignStore):
    """Campaigns newest first, with the default leading; broken bundles are skipped."""

    def campaigns(self) -> list[dict]:
        found = []
        if self.root.exists():
            for path in sorted(self.root.iterdir()):
                if path.name.startswith(".") or not (path / "manifest.json").is_file():
                    continue
                try:
                    found.append(self.campaign(path.name))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    # One broken bundle must not hide every other campaign.
                    warnings.warn(f"Skipping campaign {path.name}: {error}", stacklevel=2)
        found.sort(key=lambda m: m.get("generated_at") or "", reverse=True)
        found.sort(key=lambda m: m["campaign_id"] != DEFAULT_CAMPAIGN)
        return found

    def default_campaign(self, campaigns: list[dict] | None = None) -> str | None:
        ids = [m["campaign_id"] for m in (self.campaigns() if campaigns is None else campaigns)]
        return DEFAULT_CAMPAIGN if DEFAULT_CAMPAIGN in ids else (ids[0] if ids else None)
