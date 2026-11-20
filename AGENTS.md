# AGENTS.md

Instructions for coding agents (Codex, Claude Code) working on this repo. Read README.md for what the tool
does and how it is used.

## The product in one paragraph

A local Python tool that finds highlight moments in GTA VI stream VODs (audio, transcript, chat, Twitch
viewer clips), lets a human review and label them, tunes its weights on those labels, and renders vertical
clips with captions, one honest version per platform (TikTok, Instagram Reels, YouTube Shorts). The owner
posts them by hand for clipping campaigns (Content Rewards / Whop). Revenue depends on campaigns accepting
the clips, so rights, disclosure and no duplicates matter as much as detection quality.

## Rules that are not up for discussion

1. **Rights.** Every source has a `rights` note (`sources.rights` is NOT NULL). Never add a way to ingest or
   render without it, and keep it in each export's `.json`.
2. **No unofficial posting.** No browser automation, no reverse-engineered or private endpoints, no scraping
   uploads, no tools for several accounts per platform. Publishing work uses official APIs only (YouTube Data
   API, Instagram Graph API, TikTok Content Posting API once the app is audited), always behind an explicit
   command run by the user.
3. **No tricks against platform detection.** Platform versions differ by cut points, zoom (8% at most) and
   caption colour. Never add noise, mirroring, speed or pitch changes, invisible overlays or metadata games
   meant to get past Content ID or duplicate detection.
4. **Disclosure.** The "PUBLICITÉ" tag option and the metadata `.json` stay. Paid campaign clips must be
   labelled.
5. **Duplicate guard.** Never weaken `clipper/dedup.py` or the checks in `pipeline.render_candidates` to make
   a test pass. Exports recorded as posted are never replaced or deleted.
6. **No gambling.** No support for casino, betting or loot-box campaigns.
7. **Tests stay offline and deterministic.** No network, no real VODs, no model downloads. Tests use the
   synthetic stream from `clipper/fixture.py`, which is deterministic (fixed seeds): keep it that way.

## Code map

| path | job |
|---|---|
| `clipper/cli.py` | the `clipper` command (argparse): doctor, demo, ingest, analyze, layout, render, exports, posted, label, eval, tune, review |
| `clipper/pipeline.py` | orchestration: ingest, cached signals, analyze, layout, render with the duplicate guard |
| `clipper/signals/` | one module per signal; each returns a curve on the 0.5 s timeline |
| `clipper/detect.py` | weighted sum of signals, peak picking, cut snapping (speech pauses), overlap removal |
| `clipper/evaluate.py` | precision/recall@k against labels, random-search weight tuning |
| `clipper/layout.py` | facecam detection (YuNet + overlay border), layout choice, preview PNG |
| `clipper/render.py` | FFmpeg filter graphs, encoders, loudness, covers, preview proxies |
| `clipper/subtitles.py` | ASS captions: word highlight, hook box, disclosure tag |
| `clipper/variants.py` | one version per platform (seeded, distinct cut points) |
| `clipper/dedup.py` | perceptual signatures of the source gameplay, time overlap checks |
| `clipper/db.py` | SQLite: sources, candidates, labels, exports, posts |
| `clipper/config.py` | YAML layers: defaults, local, streamer profile, tuned weights, env, extra file |
| `clipper/review_app.py` | Streamlit review app |
| `clipper/defaults.yaml` | every tunable, with comments |
| `clipper/fixture.py` | deterministic synthetic stream used by tests and `clipper demo` |
| `tests/` | pytest; `conftest.py` builds the fixture once per session |

## Commands

```bash
pip install -e ".[transcribe,review,dev]"
pytest -q                  # full suite, about 2 minutes on 2 CPUs (renders small clips)
pytest -q -m "not slow"    # quick loop while coding
clipper doctor             # environment check
clipper demo               # end-to-end run on the synthetic stream; must print precision@3 = 1.0
```

## Conventions

- Python 3.10+, type hints, `from __future__ import annotations`. Keep functions small and pure where you can;
  side effects live in `pipeline.py` and `cli.py`.
- FFmpeg and ffprobe only through `clipper.media` (`media.run`, `media.ffmpeg_bin()`), as argument lists.
  Never `shell=True`.
- Windows first: paths can have spaces and backslashes. Filters that take file names (`subtitles`) run with
  `cwd` set to the work folder and relative names, as in `render.py`. Do not put absolute Windows paths
  inside filter strings.
- Settings: every tunable goes in `clipper/defaults.yaml` with a comment and is read with
  `config.get(cfg, "a.b.c", default)`. No magic numbers in code. Lists in YAML layers replace, not merge.
- A new signal: a module in `clipper/signals/`, its name in `detect.SIGNAL_NAMES`, a weight in
  `detection.weights`, wiring in `pipeline.compute_signals` (cache anything slow in `raw_signals`), tests on
  the fixture. `evaluate.tune` and the review app pick it up from `SIGNAL_NAMES`.
- Database: additive changes only (`CREATE TABLE IF NOT EXISTS`, new nullable columns). Users have data.
- Files from users are read with `encoding="utf-8-sig"` (Excel and PowerShell add a BOM) and written as
  UTF-8.
- Language: code, comments, CLI messages and docs in English; captions and hooks in French (the audience).
- Look at what you render. Exit code 0 does not mean the clip looks right: grab a frame
  (`ffmpeg -ss 5 -i clip.mp4 -frames:v 1 frame.png`) and check it.

## Two agents on one repo

- **Claude Code** builds: implementation and its tests, on a branch `feat/<topic>`.
- **Codex** reviews: reads the diff, adds tests for edge cases, fixes what the review finds, on a branch
  `review/<topic>` from the feature branch.
- Each agent works in its own git worktree so they never edit the same files at once:
  `git worktree add ../clipper-review review/<topic>`.
- `TASKS.md` is the shared log. Take one task, mark it `in progress (claude)` or `in progress (codex)`, and
  when it is merged move it to Done with the date and one line on what changed.

## Repository skills

Use [clipper-verify](.agents/skills/clipper-verify/SKILL.md) for a synthetic end-to-end verification and [clipper-review](.agents/skills/clipper-review/SKILL.md) for a branch or working-tree audit.

## Definition of done

- `pytest -q` green, `clipper doctor` ready, `clipper demo` still at precision@3 = 1.0.
- New behaviour has tests; anything that touches rendering has one end-to-end test that renders a clip.
- New settings documented in `defaults.yaml`; README updated when the user's workflow changes.
- `TASKS.md` updated.
