"""Browser regression checks with deterministic API fixtures and no external requests.

Run a Vite dev/preview server, then:
    uv run python web/tests/ui_smoke.py --url http://127.0.0.1:5175
Uses installed Chrome by default; --channel chromium uses Playwright's browser.
"""

from __future__ import annotations

import argparse
import copy
import re
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright


def fixtures():
    status = {
        "board_available": True,
        "built_at": "2026-09-01",
        "player_count": 3,
        "metric_versions": {
            version: {
                "available": True,
                "player_count": 3,
                "built_at": "2026-09-01T12:00:00Z",
                "forecast_as_of": ["2026-08-31"],
                "provenance": {
                    "state": "frozen_reference" if version == "v1" else "stale",
                    "artifact_id": "fixture-artifact",
                },
            }
            for version in ("v1", "v2", "adaptive")
        },
        "metric_report": {"available": True, "built_at": "2026-09-01"},
        "league": {
            "name": "Fixture League",
            "team_count": 2,
            "board_season": 2025,
            "draft_season": 2026,
            "seasons": [2023, 2024, 2025],
            "vor_baseline_rank": {},
        },
    }
    players = []
    for index, name in enumerate(("Alpha Receiver", "Beta Receiver", "Missing Receiver")):
        players.append(
            {
                "player_id": str(index),
                "player_display_name": name,
                "metric_version": "v2",
                "rank": index + 1,
                "position": "WR",
                "team": "BUF",
                "espn_team": "BUF",
                "espn_draft_rank": [40, 1, 2][index],
                "adj_vor": 4 - index,
                "vor": 4 - index,
                "ppg": 8,
                "games": 12,
                "season_pts": 96,
                "bonus_pts": 0,
                "flags": "",
                "floor": 4,
                "volatility": 2,
                "forecast_active_ppg": [10, 20, None][index],
                "forecast_expected_games": [10, 5, None][index],
                "forecast_season_points": [100, 100, None][index],
                "fitted_ppg": [90, 1, 999][index],
                "fitted_games": 17,
                "fitted_season_points": [1530, 17, 9999][index],
                "proj_ppg": 888,
                "v2_overall_vor": 4 - index,
                "v2_position_rank": index + 1,
                "v2_rank_key": "fitted_season_points",
                "v2_rank_vor": 4 - index,
                "forecast_source": "production_returner_v1",
                "forecast_as_of": "2026-08-31",
                "forecast_status": "unavailable" if index == 2 else "preseason",
                "rank_source": "model",
                "availability": "rostered",
                "is_mine": index == 0,
                "owner_team_name": "Team 1" if index == 0 else "Team 2",
                "injury_status": "ACTIVE",
                "changed_team": False,
                "wire_vor": 1 - index,
            }
        )
    teams = []
    for index in range(2):
        teams.append(
            {
                "team_id": index + 1,
                "team_name": f"Team {index + 1}",
                "is_mine": index == 0,
                "owner": "Manager",
                "wins": 1,
                "losses": 1,
                "faab_remaining": 50,
                "division_id": None,
                "division_name": None,
                "team_score": 5,
                "ranking_metric": "v2_overall_vor",
                "ranking_total": 10,
                "expected_lineup_vor": None,
                "lineup_vor_risk": None,
                "risk_adjusted_total": None,
                "availability_floor_points": 80,
                "availability_spread_points": 12,
                "team_rank": index + 1,
                "scored_players": 3,
                "fallback_players": 0,
                "expected_weekly_points": 100,
                "weekly_risk": 10,
                "weekly_floor": 90,
                "lineup_coverage": 0.9,
                "bench_rescue_points": 3,
            }
        )

    def ranking(name, role, hits, ndcg):
        return {
            "ranker": name,
            "role": role,
            "window": "modern",
            "window_label": "Modern",
            "position": "ALL",
            "target": "actual_availability_value",
            "k": 2,
            "pool": 4,
            "folds": 2,
            "hit_rate": sum(hits) / 2,
            "ndcg": 0.6,
            "pool_spearman": 0.5,
            "top_k_actual_mean": 1,
            "ideal_top_k_actual_mean": 2,
            "hit_rate_lift": 0.125 if role == "candidate" else None,
            "hit_rate_lift_ci_low": -0.1,
            "hit_rate_lift_ci_high": 0.3,
            "best_baseline": "market_ecr_score",
            "beats_baseline": False,
            "fold_results": [
                {
                    "forecast_season": year,
                    "n": 4 if role == "candidate" else 6,
                    "outcome_n": 8,
                    "coverage": 0.5,
                    "missing_scores": 4,
                    "hit_rate": hits[i],
                    "ndcg": ndcg[i],
                    "pool_spearman": 0.4,
                    "top_k_actual_mean": 1,
                    "ideal_top_k_actual_mean": 2,
                }
                for i, year in enumerate((2024, 2025))
            ],
        }

    rankings = [
        ranking("fitted_season_points", "candidate", [0.75, 0.5], [0.8, None]),
        ranking("fitted_games", "candidate", [0.5, 0.25], [0.5, 0.3]),
        ranking("market_ecr_score", "baseline", [0.5, 0.5], [0.4, 0.6]),
    ]
    windows = [
        {"key": "long_horizon", "label": "Long horizon", "start": 2004, "end": 2025},
        {"key": "modern", "label": "Modern enriched", "start": 2024, "end": 2025},
        {"key": "recent", "label": "Recent era", "start": 2025, "end": 2025},
        {"key": "market", "label": "Market archive", "start": 2025, "end": 2025},
        {"key": "market_full", "label": "Market incl. backfill", "start": 2011, "end": 2025},
    ]
    report = {
        "title": "Fixture research",
        "generated_at": "2026-09-01T12:00:00Z",
        "configuration": {
            "history_seasons": 3,
            "analysis_windows": windows,
            "ranking": {"targets": ["actual_availability_value"]},
        },
        "data_summary": {
            "completed_forecasts": [2024, 2025],
            "pending_forecasts": [2026],
            "forecast_rows": 6,
            "completed_rows": 4,
            "limitations": [],
        },
        "metrics": [],
        "results": [],
        "model_results": [],
        "uncertainty_calibration": [
            {
                "position": "WR",
                "population": "returner",
                "forecast_season": 2025,
                "n_train": 100,
                "n_test": 50,
                "nominal_coverage": 0.8,
                "observed_coverage": 0.82,
                "residual_lower": -40.0,
                "residual_upper": 50.0,
                "status": "report_only",
            }
        ],
        "targets": [],
        "ranking_results": [
            {
                **row,
                "window": window["key"],
                "fold_results": [
                    fold
                    for fold in row["fold_results"]
                    if window["start"] <= fold["forecast_season"] <= window["end"]
                ],
            }
            for window in windows
            for row in rankings
        ],
        "population_ranking_results": [
            {
                **row,
                "population": "rookie",
                "ranker": "fitted_nextgen_season_points"
                if row["role"] == "candidate"
                else row["ranker"],
            }
            for row in rankings
            if row["ranker"] != "fitted_games"
        ],
        "fitted_model_summary": [
            {
                "model": "fitted_ppg",
                "position": "WR",
                "latest_forecast_season": 2026,
                "coefficients": {"historical_ppg_prior": 0.5},
                "coefficient_sd_across_refits": {"historical_ppg_prior": 0.1},
                "n_train": 100,
                "refits": 2,
                "ridge_lambda": 0.1,
            },
            {
                "model": "adaptive_selector",
                "position": "WR",
                "latest_forecast_season": 2026,
                "kind": "adaptive_select",
                "selected_source": "fitted_season_points",
                "selection_folds": 2,
                "selection_score": 0.625,
                "selected_source_counts": {"fitted_season_points": 2},
                "refits": 2,
            },
        ],
    }
    return status, players, teams, report


def run(url: str, channel: str):
    status, players, teams, report = fixtures()
    state = {"connected": True}
    freshness = {"age_seconds": 30, "stale": False, "week": 1, "season": 2026}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=channel if channel != "chromium" else None)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.set_default_timeout(10000)
        errors = []
        league_versions = []
        research_requests = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def route_api(route):
            parsed = urlparse(route.request.url)
            path = parsed.path
            version = parse_qs(parsed.query).get("version", ["v2"])[0]
            dataset = parse_qs(parsed.query).get("dataset", [None])[0]
            if path in {
                "/api/research/catalog",
                "/api/research/players",
                "/api/research/current-players",
                "/api/research/league-impact",
                "/api/research/report",
            }:
                research_requests.append((path, dataset))
            if path.startswith("/api/league") and path != "/api/league/status":
                league_versions.append(version)
            rows = copy.deepcopy(players)
            for row in rows:
                row["metric_version"] = version
                if version == "adaptive":
                    row.update(
                        forecast_season_points=200 - row["rank"],
                        forecast_source="frozen_adaptive",
                        forecast_status="frozen_experiment",
                    )
            if path == "/api/status":
                payload = status
            elif path == "/api/research/sources":
                payload = {
                    "default_id": "repaired_fixture",
                    "unavailable": [],
                    "datasets": [
                        {
                            "id": name,
                            "label": name,
                            "accepted": name != "legacy",
                            "accepted_at": "2026-09-22",
                            "checks_passed": 12,
                            "forecast_rows": 18289,
                            "report_available": True,
                            "description": "Research only; production unchanged.",
                        }
                        for name in ["repaired_fixture", "legacy"]
                    ],
                }
            elif path == "/api/research/catalog":
                payload = {
                    "saved_at": "2026-09-01",
                    "seasons": [
                        {"season": 2026, "complete": False, "cutoff_dates": ["2026-08-28"]},
                        {"season": 2025, "complete": True},
                    ],
                    "models": [
                        {
                            "id": m,
                            "seasons": [2025, 2026],
                            "unit": "games" if m == "fitted_games" else "season points",
                        }
                        for m in [
                            "fitted_season_points",
                            "fitted_nextgen_season_points",
                            "fitted_games",
                        ]
                        + (["frozen_adaptive_2026"] if dataset == "legacy" else [])
                    ],
                }
            elif path == "/api/research/league-impact":
                weight = float(parse_qs(parsed.query).get("wr", ["1"])[0])
                payload = {
                    **freshness,
                    "saved_at": "2026-09-01",
                    "basis": "Temporary best legal lineup scenario.",
                    "teams": [
                        {
                            "team_id": i + 1,
                            "team_name": f"Team {i + 1}",
                            "baseline_rank": i + 1,
                            "scenario_rank": i + 1 if weight == 1 else 2 - i,
                            "baseline_value": 10,
                            "scenario_value": 10 * weight,
                            "value_change": 10 * weight - 10,
                            "rank_change": 0,
                            "wins": 1,
                            "losses": 1,
                            "modeled_players": 2,
                            "fallback_players": 0,
                            "missing_players": 0,
                            "starters": [],
                        }
                        for i in range(2)
                    ],
                }
            elif path in {"/api/research/players", "/api/research/current-players"}:
                payload = {
                    "season": 2026,
                    "model": "fitted_season_points",
                    "model_unit": "season points",
                    "saved_at": "2026-09-01",
                    "pool_size": 3,
                    "scored_players": 2,
                    "complete": False,
                    "ranking_basis": "Search does not change ranks.",
                    "outcome_status": "Current league observations are partial."
                    if path.endswith("current-players")
                    else "Saved outcomes.",
                    "players": [
                        {
                            **r,
                            "model_rank": i + 1 if i < 2 else None,
                            "model_value": 100 - i if i < 2 else None,
                            "actual_rank": 3 - i,
                            "actual_value": i * 10,
                            "benchmark_rank": i + 1,
                            "benchmark_value": 10,
                            "rank_gap": 2 - 2 * i,
                            "population": "returner",
                        }
                        for i, r in enumerate(rows)
                    ],
                }
            elif path == "/api/profiles/player":
                from patron.metrics.player_profile import METRICS

                player = next(
                    r for r in rows if r["player_id"] == parse_qs(parsed.query)["player_id"][0]
                )
                payload = {
                    "version": "profiles_fixture",
                    "identity": {**player, "rookie_season": 2026},
                    "cutoff": {"season": 2026, "week": 2},
                    "available_through": {"season": 2026, "week": 2},
                    "sources": {
                        "history": "repaired_fixture",
                        "college": "college_fixture",
                        "outlook": "outlook_fixture",
                        "observations_saved_at": "2026-09-22",
                    },
                    "links": [],
                    "college_seasons": [],
                    "nfl_seasons": [],
                    "career": {},
                    "features": {},
                    "role_history": [],
                    "role_periods": [],
                    "transitions": [],
                    "injury_reports": [],
                    "preseason_forecasts": [],
                    "college_forecasts": [],
                    "college_development_forecasts": [],
                    "current_outlook": None,
                    "missing_evidence": [],
                    "metrics": METRICS,
                    "limits": [],
                }
            elif path == "/api/profiles":
                payload = {
                    "report": {"season": 2026, "through_week": 2},
                    "total": len(rows),
                    "players": rows,
                }
            elif path == "/api/research/outlook":
                payload = {
                    "version": "outlook_fixture",
                    "season": 2026,
                    "through_week": 2,
                    "horizon": [3, 6],
                    "observations_saved_at": "2026-09-22",
                    "saved_at": "2026-09-22",
                    "history": {"version": "repaired_fixture"},
                    "observation_age_hours": 1,
                    "newer_week_possible": False,
                    "ownership_available": True,
                    "ownership_stale": False,
                    "confidence_scores_published": False,
                    "validation": [],
                    "limitations": [],
                    "design": {"method": "Earlier seasons only", "publication_policy": "Gated"},
                    "players": [
                        {
                            **row,
                            "position_rank": i + 1,
                            "population": "rookie",
                            "eligible": i == 0,
                            "forecast_next4": 25 if i == 0 else None,
                            "usage_baseline_next4": 24 if i == 0 else None,
                            "points_pace_next4": 20 if i == 0 else None,
                            "recent_targets_pg": 3,
                            "recent_carries_pg": 0,
                            "recent_points_pg": 5,
                            "recent_snap_share": 0.5,
                            "past_team_games": 2,
                            "future_team_games": 4,
                            "prior_offensive_weeks": 0,
                            "observed_offensive_weeks": 2,
                            "snap_observations": 2,
                            "snap_feed_complete": True,
                            "opportunity_trend": "insufficient history",
                            "role_evidence": "insufficient history",
                            "snap_change": None,
                            "snap_std": None,
                            "points_std": None,
                            "spike_share": None,
                            "bonus_share": 0,
                            "usage_change": None,
                            "expected_offensive_weeks": 3.2,
                            "participation_pace_next4": 4,
                            "expected_active_snap_share": 0.5,
                            "outcome_range": None,
                            "range_status": "Withheld: gates not passed",
                            "confidence_score": None,
                            "forecast_status": "Exploratory outlook",
                        }
                        for i, row in enumerate(rows[:2])
                    ],
                }
            elif path == "/api/research/rookies":
                payload = {
                    "season": 2026,
                    "through_week": 2,
                    "horizon": [3, 6],
                    "saved_at": "2026-09-22",
                    "forecast_age_hours": 1,
                    "newer_week_possible": False,
                    "ownership_available": True,
                    "ownership_stale": False,
                    "ownership_captured_at": "2026-09-22",
                    "history": {"version": "repaired_fixture"},
                    "current_source": "Saved NFL observations",
                    "method": "Earlier rookie analogs",
                    "limitations": ["Exploratory, not trade value."],
                    "backtest": {"basis": "Earlier seasons only", "positions": []},
                    "players": [
                        {
                            **row,
                            "position_rank": i + 1,
                            "draft_pick": 20 + i,
                            "points_to_date": 10,
                            "targets": 5,
                            "carries": 0,
                            "snap_share": 0.4,
                            "observed_stat_weeks": 2,
                            "snap_observations": 2,
                            "forecast_next4": 25 if i == 0 else None,
                            "pace_next4": 20,
                            "analog_p10": 5,
                            "analog_p90": 45,
                            "history_count": 30,
                            "neighbor_count": 10,
                            "forecast_status": "Exploratory analog estimate",
                            "latest_targets": 3,
                            "latest_carries": 0,
                            "analogs": [],
                        }
                        for i, row in enumerate(rows[:2])
                    ],
                }
            elif path in {"/api/metric-report", "/api/research/report"}:
                payload = report
            elif path == "/api/league/status":
                payload = {"join": None}
            elif path == "/api/league":
                payload = {**freshness, "teams": teams}
            elif path == "/api/league/players" and not state["connected"]:
                route.fulfill(status=503, json={"detail": "Fixture: not connected"})
                return
            elif path in {"/api/league/players", "/api/board"} or path.startswith(
                "/api/league/roster/"
            ):
                payload = {**freshness, "players": rows, "total": len(rows)}
            elif path == "/api/league/wire":
                payload = {**freshness, "players": rows, "replacement_levels": {"WR": 10}}
            elif path == "/api/league/compare":
                side = {
                    "source": "projected",
                    "metric": "v2_overall_vor",
                    "total": 10,
                    "starters": [],
                    "bench": [],
                    "unranked_starters": 0,
                }
                payload = {
                    "left": {**side, "team_name": "Team 1"},
                    "right": {**side, "team_name": "Team 2"},
                    "margin": 0,
                }
            else:
                payload = {**freshness, "teams": [], "players": [], "transactions": [], "total": 0}
            route.fulfill(json=payload)

        page.route(re.compile(r"^https?://[^/]+/api/"), route_api)
        page.goto(url)
        expect(page.get_by_role("tablist", name="Section").get_by_role("tab")).to_have_text(
            ["01 League", "02 Intelligence"]
        )
        expect(page.get_by_role("tablist", name="Metric basis")).to_have_count(0)
        expect(page.get_by_role("heading", name="Standings", exact=True)).to_be_visible()
        page.get_by_role("button", name="Show simulation breakdown").click()
        expect(page.get_by_role("columnheader", name="Availability downside")).to_be_visible()
        expect(page.get_by_role("columnheader", name="Scenario VOR")).to_have_count(0)
        page.screenshot(path="/tmp/league-desktop.png", full_page=True)
        page.get_by_role("tab", name="League", exact=True).click()
        page.get_by_role("tab", name="Players", exact=True).click()
        expect(page.get_by_text("Stale artifact", exact=True)).to_be_visible()
        page.get_by_role("button", name="Active PPG", exact=True).click()
        expect(page.locator("tbody tr").first).to_contain_text("Beta Receiver")
        expect(page.locator("tbody tr").last).to_contain_text("Missing Receiver")
        expect(page.locator("tbody tr").last).not_to_contain_text("999")
        page.get_by_role("button", name="Show details for Alpha Receiver").click()
        expect(
            page.get_by_role("heading", name="Alpha Receiver · forecast and evidence")
        ).to_be_visible()
        page.get_by_text("Research inputs and projection assumptions", exact=True).click()
        expect(page.get_by_text("888.0", exact=True)).to_be_visible()
        expect(page.get_by_text("Dropback on-field share (proxy)", exact=True)).to_be_visible()
        page.get_by_role("tab", name="Intelligence", exact=True).click()
        expect(
            page.get_by_role("tablist", name="Intelligence views").get_by_role("tab")
        ).to_have_text(["Players", "League impact", "Research"])
        page.get_by_role("tab", name="League impact", exact=True).click()
        expect(page.get_by_label("Research dataset")).to_have_value("repaired_fixture")
        expect(page.get_by_text("12 integrity checks passed", exact=False)).to_be_visible()
        expect(
            page.get_by_label("Research model").locator('option[value="fitted_games"]')
        ).to_have_count(0)
        page.get_by_label("WR weight (%)").fill("150")
        expect(page.locator("tbody tr").first).to_contain_text("Team 2")
        page.get_by_role("button", name="Reset assumptions").click()
        expect(page.locator("tbody tr").first).to_contain_text("Team 1")
        page.get_by_role("tab", name="Players", exact=True).click()
        page.get_by_label("Player view").select_option("players")
        page.get_by_label("Model list").select_option("all")
        expect(
            page.get_by_label("Research model").locator('option[value="fitted_games"]')
        ).to_have_count(1)
        page.get_by_label("Model list").select_option("player")
        page.get_by_label("Population", exact=True).select_option("rookie")
        expect(page.get_by_label("Research model")).to_have_value("fitted_nextgen_season_points")
        page.get_by_label("Population", exact=True).select_option("all")
        page.get_by_label("Research model").select_option("fitted_season_points")
        expect(page.get_by_role("columnheader", name="Current roster")).to_be_visible()
        expect(page.locator("tbody tr.mine-row")).to_contain_text("Alpha Receiver")
        expect(page.locator("tbody tr.mine-row")).to_contain_text("Team 1 · Mine")
        page.get_by_label("Current roster", exact=True).select_option("mine")
        expect(page.locator("tbody tr")).to_have_count(1)
        page.get_by_label("Current roster", exact=True).select_option("all")
        page.get_by_label("Search players", exact=True).fill("Beta")
        expect(page.locator("tbody tr")).to_have_count(1)
        expect(page.locator("tbody tr")).to_contain_text("#2")
        page.get_by_label("Search players", exact=True).fill("")
        page.get_by_label("Research dataset").select_option("legacy")
        page.get_by_label("Research model").select_option("frozen_adaptive_2026")
        page.get_by_label("Reality source").select_option("current")
        expect(
            page.get_by_text("Current league observations are partial.", exact=True)
        ).to_be_visible()
        page.get_by_role("tab", name="Research", exact=True).click()
        page.get_by_label("Research topic").select_option("mispricing")
        expect(page.locator("tbody tr")).to_have_count(1)
        expect(page.locator("tbody tr")).to_contain_text("Higher by 39")
        expect(page.locator("tbody tr.mine-row")).to_contain_text("Team 1 · Mine")
        page.get_by_label("Minimum rank gap", exact=True).fill("0")
        expect(page.locator("tbody tr")).to_have_count(2)
        expect(page.locator("tbody tr").first).to_contain_text("Alpha Receiver")
        page.get_by_label("Research view", exact=True).select_option("lower")
        expect(page.locator("tbody tr")).to_contain_text("Beta Receiver")
        page.get_by_label("Research view", exact=True).select_option("all")
        page.get_by_label("Compare against", exact=True).select_option("actual")
        expect(page.locator("tbody tr").first).to_contain_text("Higher by 2")
        page.get_by_label("Compare against", exact=True).select_option("espn")
        page.screenshot(path="/tmp/intelligence-desktop.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
        page.screenshot(path="/tmp/mispricing-mobile.png", full_page=True)
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.get_by_role("tab", name="Research", exact=True).click()
        page.get_by_label("Research topic").select_option("evidence")
        page.get_by_label("Research dataset").select_option("repaired_fixture")
        expect(
            page.get_by_role("heading", name="Compare performance across seasons")
        ).to_be_visible()
        expect(page.get_by_label("Evidence window", exact=True).locator("option")).to_have_count(3)
        expect(page.get_by_label("Evidence window", exact=True)).to_have_value("modern")
        expect(page.get_by_text("These are overlapping date windows", exact=False)).to_be_visible()
        page.get_by_label("Evidence window list").select_option("all")
        expect(page.get_by_label("Evidence window", exact=True).locator("option")).to_have_count(5)
        page.get_by_label("Evidence window", exact=True).select_option("recent")
        expect(page.locator(".research-explorer .season-picker button")).to_have_count(1)
        page.get_by_label("Evidence window list").select_option("standard")
        expect(page.get_by_label("Evidence window", exact=True)).to_have_value("modern")
        expect(page.locator(".research-explorer .season-picker button")).to_have_count(2)
        page.get_by_text("Dataset quality & what to keep", exact=True).click()
        expect(
            page.get_by_text("12 integrity checks passed for repaired_fixture", exact=False)
        ).to_be_visible()
        page.get_by_text("Dataset quality & what to keep", exact=True).click()
        picker = page.get_by_label("Model to inspect", exact=True)
        expect(picker.locator("option")).to_have_text(["Core season forecast · returners"])
        page.get_by_label("Evidence model list").select_option("all")
        picker.select_option("fitted_games")
        expect(picker).to_have_value("fitted_games")
        expect(page.locator(".chart-key.model")).to_have_text("Core expected games · component")
        page.get_by_label("Evidence model list").select_option("player")
        expect(picker).to_have_value("fitted_season_points")
        page.get_by_label("Population", exact=True).select_option("rookie")
        expect(picker).to_have_value("fitted_nextgen_season_points")
        expect(page.locator(".chart-key.model")).to_have_text(
            "Next-gen season forecast · rookies + returners"
        )
        page.get_by_label("Population", exact=True).select_option("all")
        expect(page.get_by_label("Weight model", exact=True).locator("option")).to_have_text(
            ["Core active-game PPG · returners"]
        )
        calibration = page.get_by_role("region", name="Historical uncertainty calibration")
        expect(calibration.get_by_text("82.0%", exact=True).first).to_be_visible()
        expect(calibration.get_by_text("80.0%", exact=True)).to_be_visible()
        page.get_by_text("Original preseason draft list", exact=True).click()
        expect(page.get_by_role("columnheader", name="Historical PPG", exact=True)).to_be_visible()
        page.get_by_role("tab", name="Players", exact=True).click()
        page.get_by_label("Population", exact=True).select_option("rookie")
        page.get_by_label("Player view").select_option("outlook")
        expect(page.get_by_label("Research dataset")).to_have_count(0)
        page.get_by_label("Position", exact=True).select_option("WR")
        expect(page.get_by_role("columnheader", name="80% outcome range")).to_be_visible()
        expect(page.locator("tbody tr").first).to_contain_text("25.0 / 24.0")
        expect(page.locator("tbody tr").first).to_contain_text("Withheld")
        page.get_by_label("Outlook coverage").select_option("missing")
        expect(page.locator("tbody tr")).to_have_count(1)
        expect(page.locator("tbody tr")).to_contain_text("Beta Receiver")
        page.get_by_label("Outlook coverage").select_option("scored")
        page.get_by_role("button", name="Show details for Alpha Receiver").click()
        expect(page.get_by_role("heading", name="Alpha Receiver", exact=True)).to_be_visible()
        page.get_by_role("button", name="Forecasts", exact=True).click()
        expect(page.get_by_role("heading", name="Historical rookie comparisons")).to_be_visible()
        expect(page.get_by_text("Latest completed week: 3 targets", exact=False)).to_be_visible()
        page.get_by_text("Four-week outlook diagnostics", exact=True).click()
        expect(page.get_by_text("not a validated “flashy player”", exact=False)).to_be_visible()
        expect(page.get_by_text("needs ≥3 observations", exact=False)).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth"), (
            "Outlook overflows mobile"
        )
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.get_by_role("tab", name="League", exact=True).click()
        page.get_by_role("tab", name="View roster", exact=True).click()
        expect(page.get_by_role("columnheader", name="Active PPG", exact=True)).to_be_visible()
        expect(page.locator(".comparison-workspace tbody tr").first).to_contain_text("10.0")
        page.get_by_role("tab", name="Waiver Wire", exact=True).click()
        expect(page.get_by_role("columnheader", name="Season points", exact=True)).to_be_visible()
        expect(page.locator("tbody tr").first).to_contain_text("100")
        page.get_by_role("tab", name="Intelligence", exact=True).click()
        page.get_by_role("tab", name="Research", exact=True).click()
        page.get_by_label("Research topic").select_option("evidence")
        expect(page.get_by_text("Legacy comparison:", exact=False).first).to_be_visible()
        page.get_by_role("button", name="2024", exact=True).click()
        expect(page.locator(".fold-focus")).to_contain_text("Model 75.0%")
        page.get_by_label("Compare by", exact=True).select_option("ndcg")
        expect(page.locator(".research-explorer .research-kpis").first).to_contain_text(
            "1 shared season"
        )
        expect(page.locator(".research-explorer .research-kpis").first).to_contain_text("0.400")
        page.get_by_label("Weight model list").select_option("all")
        page.get_by_label("Weight model", exact=True).select_option("adaptive_selector")
        expect(page.get_by_text("Historical source selections", exact=True)).to_be_visible()
        page.get_by_text("Exact fold values and coverage (2 seasons)", exact=True).click()
        with page.expect_download() as download:
            page.get_by_role("button", name="Download fold CSV").click()
        with open(download.value.path()) as file:
            csv = file.read()
        assert '"2024","fitted_season_points","market_ecr_score","4","6","0.75"' in csv
        page.get_by_label("Population", exact=True).select_option("rookie")
        expect(
            page.get_by_role("heading", name="Compare performance across seasons")
        ).to_be_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth"), (
            "Research overflows mobile"
        )
        page.screenshot(path="/tmp/intelligence-mobile.png", full_page=True)
        page.get_by_role("tab", name="League", exact=True).click()
        page.get_by_role("tab", name="Players", exact=True).click()
        page.get_by_role("button", name="Show details for Alpha Receiver").click()
        assert not page.evaluate("document.documentElement.scrollWidth > innerWidth"), (
            "Player details overflow mobile"
        )
        state["connected"] = False
        page.reload()
        page.get_by_role("tab", name="League", exact=True).click()
        page.get_by_role("tab", name="Players", exact=True).click()
        expect(page.get_by_text("Showing the metric board only", exact=False)).to_be_visible()
        page.get_by_label("Search players", exact=True).fill("Nobody")
        expect(page.get_by_text("No players match these filters.", exact=True)).to_be_visible()
        component = next(
            row
            for row in report["ranking_results"]
            if row["ranker"] == "fitted_games" and row["window"] == "modern"
        )
        report["ranking_results"] = []
        report["population_ranking_results"] = []
        page.reload()
        page.get_by_role("tab", name="Intelligence", exact=True).click()
        page.get_by_role("tab", name="Research", exact=True).click()
        page.get_by_label("Research topic").select_option("evidence")
        expect(
            page.get_by_text("No ranking results for this window and population.", exact=True)
        ).to_be_visible()
        report["ranking_results"] = [{**component, "window": "market"}]
        report["configuration"]["analysis_windows"] = [
            {"key": "market", "label": "Market archive", "start": 2024, "end": 2025}
        ]
        report["fitted_model_summary"] = [
            row for row in report["fitted_model_summary"] if row["model"] == "adaptive_selector"
        ]
        page.reload()
        page.get_by_role("tab", name="Intelligence", exact=True).click()
        page.get_by_role("tab", name="Research", exact=True).click()
        page.get_by_label("Research topic").select_option("evidence")
        expect(page.get_by_label("Evidence window", exact=True)).to_have_value("market")
        expect(page.get_by_label("Model to inspect", exact=True)).to_be_disabled()
        expect(
            page.get_by_text("No player forecasts in this slice.", exact=False).last
        ).to_be_visible()
        page.get_by_label("Evidence model list").select_option("all")
        expect(page.get_by_label("Model to inspect", exact=True)).to_have_value("fitted_games")
        expect(page.get_by_label("Weight model", exact=True)).to_be_disabled()
        page.get_by_label("Weight model list").select_option("all")
        expect(page.get_by_label("Weight model", exact=True)).to_have_value("adaptive_selector")
        status["metric_versions"]["v2"]["available"] = False
        page.reload()
        expect(page.get_by_role("heading", name="Standings", exact=True)).to_be_visible()
        assert league_versions and set(league_versions) == {"v2"}, league_versions
        assert research_requests and all(
            dataset in {"legacy", "repaired_fixture"} for _, dataset in research_requests
        )
        assert ("/api/research/report", "repaired_fixture") in research_requests
        assert ("/api/research/current-players", "legacy") in research_requests
        assert not errors, errors
        browser.close()
    print(
        "PASS: canonical values and sorting, version resets, details, roster/wire, "
        "research filters, ownership, market comparisons, scenarios, nulls, CSV, mobile, "
        "offline board, empty/component-only reports, accepted datasets, rookie coverage, "
        "shared model labels, curated/advanced models, evidence-window fallback, quality guide"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5175")
    parser.add_argument("--channel", default="chrome")
    args = parser.parse_args()
    run(args.url, args.channel)
