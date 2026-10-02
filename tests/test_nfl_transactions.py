from datetime import date

from patron.data.nfl_transactions import (
    parse_official_transactions,
    parse_transaction_payload,
)


def test_parse_transaction_payload_extracts_date_team_and_description() -> None:
    payload = {
        "items": [
            {
                "date": "2024-08-31T07:00Z",
                "description": "Signed DB Adoree' Jackson. Released FB Jakob Johnson.",
                "team": {
                    "$ref": "http://sports.core.api.espn.com/v2/sports/football/"
                    "leagues/nfl/seasons/2024/teams/19?lang=en"
                },
            }
        ]
    }

    assert parse_transaction_payload(payload, year=2024, source_url="source") == [
        {
            "transaction_date": date(2024, 8, 31),
            "transaction_year": 2024,
            "category": "espn",
            "from_team": None,
            "source_team": "NYG",
            "to_team": None,
            "player_name": None,
            "description": "Signed DB Adoree' Jackson. Released FB Jakob Johnson.",
            "source_url": "source",
        }
    ]


def test_parse_official_transactions_uses_club_dates_and_cutoff() -> None:
    html = """
    <table><tr><td><span>08/27</span></td><td>Traded C Example.</td></tr>
    <tr><td>08/29</td><td>Placed RB Late Player on Injured Reserve.</td></tr></table>
    """

    rows = parse_official_transactions(
        html,
        year=2021,
        team="BAL",
        source_url="https://example.test/transactions/2021",
        cutoff_date=date(2021, 8, 27),
    )

    assert len(rows) == 1
    assert rows[0]["transaction_date"] == date(2021, 8, 27)
    assert rows[0]["category"] == "official"
    assert rows[0]["source_team"] == "BAL"
    assert rows[0]["to_team"] is None
