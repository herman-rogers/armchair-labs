# Career comparison coverage repair

The old comparison rule discarded an entire career when even one observed week
had a null comparison statistic. Positive offensive snaps can exist without a
box-score row, so this affected every supported position.

The descriptive comparison method now:

- Uses the same set of covered weeks for every metric in each season. A week is
  covered only when every required statistic is finite, including a recorded or
  individually verified zero.
- Requires at least eight covered weeks and at least 80% coverage of observed
  weeks in **every** compared career season, for both the target and its peers.
- Leaves unverified missing values unknown and omits those weeks from rate
  denominators. It does not silently turn missing rows into zeros.
- Reports covered/observed counts, omitted week numbers, verified zeros and
  season-specific reasons when a career still cannot be compared.
- Retains the existing same-position, first-N-seasons alignment, completed-season
  cutoff, early-history checks and minimum of five eligible peers.

The 80% floor is an explicit descriptive coverage policy, not a calibrated
confidence level. Omitting low-activity weeks can bias rates upward; the UI
discloses this and identifies the omitted weeks. These comparisons remain
descriptive and are not forecast or ranking inputs.

## Verified zero appearances

The Bills' reports describe Josh Allen taking only one offensive snap, a handoff,
in both the [2024 finale against New England](https://www.buffalobills.com/news/top-3-things-we-learned-from-bills-at-patriots-week-18)
and the [2025 finale against the Jets](https://www.buffalobills.com/news/top-3-things-we-learned-from-bills-vs-jets-week-18-x2872).
These establish zero pass attempts, passing yards, carries and rushing yards for
those appearances. The review records preserve the source URL, source date,
player ID, season/week, team, snap count and specific verified measures.

Reviews fill only nulls on a matching snap-only observation. Different snap
counts, teams, duplicate matches, recorded statistical rows or conflicting
nonzero values prevent application. Reviews do not create observations, overwrite
recorded values, or apply outside the comparison cutoff. The implementation is
position-independent and can accept similarly reviewed evidence for other players.

This is a read-time correction for descriptive comparisons. Published profile
parquets, season-history summaries, gold inputs and forecast artifacts are
unchanged; the comparison response identifies method version 2 and includes the
applied review sources.

## Published-data audit

Verified profile release: `nextgen_products_20260923_r2_profiles`.
Completed-season cutoff: 2025. Population: all 792 `current_candidate` profiles,
compared against the full historical population at their respective career stages.

| Position | Current candidates | Previously had matches | Now have matches | Restored |
| --- | ---: | ---: | ---: | ---: |
| QB | 109 | 19 | 21 | 2 |
| RB | 209 | 47 | 59 | 12 |
| WR | 304 | 56 | 94 | 38 |
| TE | 170 | 12 | 35 | 23 |
| Total | 792 | 134 | 209 | 75 |

No previously available current-player comparison becomes unavailable. The larger
eligible pool can change nearest neighbors and standardized distances.

Restored examples include Josh Allen and Jalen Hurts; Tony Pollard, Chuba Hubbard
and Tank Bigsby; CeeDee Lamb, Mike Evans, Davante Adams and Amon-Ra St. Brown;
George Kittle, Mark Andrews and Dallas Goedert.

Allen's first eight seasons now have 128/128 covered weeks, including both reviewed
zero appearances, and ten eligible peers. His nearest five are Russell Wilson,
Cam Newton, Andy Dalton, Lamar Jackson and Baker Mayfield.

Some profiles remain ineligible for separate reasons. For example, David Njoku
has only 3/4 covered weeks in 2019, below the eight-week minimum. The response now
names that season and count instead of implying that his entire history is absent.

## Validation

```sh
uv run pytest tests/test_career_similarity.py tests/test_profile_routes.py tests/test_player_profiles.py
uv run ruff check src/patron/metrics/career_similarity.py src/patron/metrics/career_stat_evidence.py tests/test_career_similarity.py web/tests/career_similarity_smoke.py
cd web
npm run build
npx oxlint src/components/SimilarCareers.tsx
cd ..
uv run python web/tests/career_similarity_smoke.py
```

Regression coverage includes all four positions, both sides of the 80% boundary,
the eight-week minimum, common metric denominators, invalid values, verified
zeros versus unknowns, conflicting evidence, historical cutoffs, small peer pools
and retained coverage diagnostics. The browser check exercises real QB/RB/WR/TE
comparisons, expanded season tables, source links, desktop/mobile layouts and an
unavailable comparison with explicit coverage reasons.

Results: 35 targeted tests pass; Ruff and the component's Oxlint check pass; the
frontend production build succeeds. Browser checks pass for Josh Allen, Tank
Bigsby, CeeDee Lamb and Mark Andrews at desktop/mobile widths, plus Christian
McCaffrey's insufficient-coverage explanation. Vite reports the existing local
Node 21 version warning, but completes the build successfully.
