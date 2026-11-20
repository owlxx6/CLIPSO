# Production hardening review · 20 November 2026

[Pull request #1](https://github.com/owlxx6/CLIPSO/pull/1) targets `main` from `fix/production-hardening`. Remote HEAD was `bc7260bfb94c218f3741bf29c5363dfef6017733`. The branch was created in a separate worktree and carries the reviewed November 20 code changes from `6690c1c`. Tests, reproduction probes and the demo used disposable databases and synthetic media.

## Reproductions and fixes

| Area | Observation | Implemented behavior |
|---|---|---|
| Source identity | The CLI wrote a label to source B for a candidate from A. An injected sidecar-copy failure left a source row. | Both label syntaxes check ownership before writing. Ingestion stages the sidecars, then promotes them inside the source transaction and cleans up on failure. |
| Transcripts | Imported JSON had no file identity. The earlier review fix automatically adopted a legacy file with unknown provenance. | Explicit imports store validated timings, a time of import and SHA-256 together. Replacing a file with different bytes, even at the same size and timestamp, disqualifies the import. Verified imports survive force; legacy inputs require confirmation. Newly generated speech caches also include the recording hash. |
| Streamed decoding | Invalid input looked like valid empty output. A synthetic source with I-frames at 0, 1.3 and 4.7 seconds yielded ten regularly spaced samples in sparse mode. | Exit status and stderr survive streaming. Partial frames, absent audio streams and empty output raise errors. A video-only source has an explicit quiet signal. Sparse frames carry FFmpeg presentation times. Regular letterbox samples use a full decode to avoid carrying a stale I-frame across a cut. |
| Highlight evaluation | A flexible candidate consumed the only valid event for a second candidate; only one of two events counted. | A rectangular Hungarian assignment maximizes cardinality first and total quality second. Tests cover reversed input, overlapping predictions, shifted boundaries and empty sets. Tuning and the review app share this evaluator. |
| Editing | Random shifts could move endpoints inside words. Length adjustment could undo detection-time snapping. | Repair split words after detection-time fitting. The platform planner scores sentence, word, quiet and supplied scene boundaries, protects context around the peak and enforces duration and previous-cut constraints. The entire interval between cuts is preserved, including silent action. |
| Campaigns | A rights note and advertising label did not establish the terms of a specific campaign. | A per-export report tracks technical validity, recording rights, rules, human approval, submission, campaign approval and payment. The CLI rechecks artifact bytes, the current caption, policy and deadline. Evidence is local, specific to an export and policy revision. Unknown requirements block submission readiness. |

Six probes run against an isolated archive of the reviewed starting code reproduced the label write, partial ingestion, silent audio/video decode errors, incorrect sparse timestamps and greedy assignment. A separate fast-start MP4 truncated within its media data produced valid early audio and video before both decoders reported failure. A seeded independent enumeration of 250 small matching problems agreed with the evaluator on both cardinality and total quality.

## Verification

| Run | Command | Result |
|---|---|---|
| Published baseline `bc7260b` | `python -m pytest -q -ra` | 64 passed, 0 failed, 0 skipped |
| Reviewed starting tree `2fc194d` | `python -m pytest -q -ra` | 78 passed, 0 failed, 0 skipped |
| Previous behavior isolated | `python -m pytest tests/test_production_hardening.py -k "label_verifies or sidecar_copy or global_assignment or streaming_reports or sparse_keyframes"` | All six selected cases failed on the reviewed starting tree |
| Feature branch, Windows | `python -m pytest -q -ra` | 127 passed, 0 failed, 0 skipped in 115.97 s |
| Environment | `python -m clipper doctor` | Ready on Windows with Python 3.13.17, FFmpeg 4.4.1-Jellyfin, libx264, libass, loudnorm, OpenCV/YuNet and Streamlit |
| Synthetic demo | `python -m clipper demo` | All 3 planted events recovered, precision@3 and recall both 1.0; six vertical exports across three platforms |
| Paid-label visual check | `python -m clipper render 1 --ids 1 --platforms tiktok --replace --no-disclosure` on a synthetic campaign | Probe measured 1080 × 1920, H.264/AAC, 30 fps, 25.00 s. Frame at 0.225 s showed PUBLICITÉ, the hook and highlighted VOIR in the stacked layout. The cover, caption and validation JSON existed. Missing campaign terms correctly blocked readiness. |
| GitHub Actions | Python 3.11 matrix, both `pytest -m "not slow"` and `pytest -m slow` | Linux and Windows both passed the [FFmpeg compatibility revision](https://github.com/owlxx6/CLIPSO/actions/runs/44714287230). See the [complete 127-test revision](https://github.com/owlxx6/CLIPSO/actions/runs/44714988736) and [PR head checks](https://github.com/owlxx6/CLIPSO/pull/1/checks) for their recorded results. |

R1–R10 were also checked through the original review probes and migration tests. These cover blank rights, enforced paid disclosures, failed and successful replacement, full-stream output validation, stable candidate and export references, edited hooks, evaluation, source/settings invalidation, SQLite migration, Unicode paths and restricted console encodings. Posted video protection has end-to-end coverage. All database work in verification used fresh scratch data.

The same synthetic export was rechecked with a fictional, verified policy. It remained blocked pending visual/workflow review. Supplying export-bound evidence made it eligible while submission, campaign approval and payment each remained false.

The first Windows Actions run caught a version-dependent FFmpeg option. Current builds use `-fps_mode passthrough` for sparse samples, while local FFmpeg 4.4 uses `-vsync 0`. The workflow runs doctor before tests and retains per-platform JUnit results. Local speech tests used supplied words or a controlled transcriber. Real Whisper inference was unavailable locally. No real footage was used in CI or on the workstation.

## Controlled GTA benchmark

1. Obtain written authorization for the **specific recordings** and platforms. Keep the source, permission reference, dated brief, chat, viewer clips, time-aligned transcript and SHA-256 in private storage. Use a fresh `CLIPPER_HOME` and `CLIPPER_DATA_DIR`. Run `clipper doctor`. Put the verified policy under `campaigns` in the local config as described in the [README](../README.md#campaign-review).
2. Ingest two distinct authorized VODs for the same streamer, one for calibration and one for holdout. Example for the first recording follows. Substitute actual paths, ID and permission reference. Omit optional sidecars you do not have.

   ```text
   clipper ingest calibration.mp4 --streamer benchmark --campaign CAMPAIGN_ID --rights "written permission reference" --chat chat.json --clips viewer-clips.json --transcript transcript.json
   clipper analyze 1 --no-transcribe
   clipper layout 1 --preview layout.png
   clipper review
   ```

3. Have a reviewer independently annotate highlight windows across the held-out footage in a `start,end` CSV before looking at predictions. Include quiet action, dialogue, missed events and false alarms; retain all labeled events in the recall denominator. Save the original configuration and base holdout output with `clipper eval 2 --truth holdout.csv --k 10` after analyzing source 2. Replace example IDs with the IDs printed by ingest.
4. Tune only on calibration labels with `clipper tune 1 --truth calibration.csv --k 10 --seed 0 --write`. Run `clipper analyze 2 --no-transcribe` again, then repeat the same holdout evaluation. Record precision@10, recall, F1, match counts, VOD duration, signal availability and wall time.
5. Freeze candidate IDs for the editing comparison. Use `clipper render 2 --ids ID --platforms PLATFORM` and `clipper campaign-check EXPORT_ID`. Have a reviewer score context, payoff, word cuts, caption timing, visible HUD/facecam, spoilers and disclosures. Record accepted/total, hands-on correction time and blocked reasons. Keep the MP4, sidecars, review evidence and rechecked report with the benchmark. A real submission and its outcome each require their own evidence.

## Phase 3 interfaces and experiments

The **Highlight Hunter** owns candidate ID, source identity, scored peak, signal evidence and protected setup/payoff ranges. The **Video Editor** consumes that candidate and produces a versioned plan. The **Orchestrator** validates the plan, enforces rights and duplicate checks, runs the existing FFmpeg renderer, and records the result for human review.

| Proposed contract | Required data and invariants |
|---|---|
| `EditPlan v1` | Recording SHA-256, source/candidate IDs, platform, policy digest, plan revision and timebase. Start with one continuous half-open source interval, in integer milliseconds. Include protected intervals, output-relative caption cues, hook, layout, disclosure and intended duration. Enforce monotonic times, bounds, protected-event inclusion and agreement between source and output duration before rendering. |
| `ExportReceipt v1` | Plan digest, attempt ID, artifact SHA-256, source SHA-256, encoder and FFmpeg versions, probed streams/durations, full-decode result, duplicate evidence, policy digest and review/outcome evidence references. Attach the receipt only after staged validation and atomic promotion. Replaying a plan must reproduce timing, visible overlays and validated streams; codec builds may produce different bytes. |

First round-trip one existing single-cut variant through a serialized plan and compare it against today's output. Then run three paired experiments using the same candidates and blinded review order. Auto-editor-inspired pacing can propose alternate edges, with protected silent action as a regression case. Kinocut-inspired receipts can add freeze/black-frame diagnostics, measuring false alarms on menus and night scenes. WowClip-inspired timelines can test optional shot-aware framing inside the facecam region while a reviewer checks HUD visibility. These are design proposals. This sprint adds no upstream runtime or new rendering engine.

## Remaining limits

- A fingerprint verifies byte identity. The importer must verify that supplied words actually describe that recording. Old generated caches can still be read using their historical path/size/mtime key. Forced regeneration upgrades their identity. Full hashing and regular frame sampling add I/O on long VODs; measure this on the first benchmark.
- A local reviewer must establish permission and the completeness of the policy, inspect any required overlays, and verify editing restrictions. Evidence references rely on trusted local files and reviewers. Each report records its check time; rerun the check before submission.
- Scene cuts can be supplied as boundary cues. The current detector measures letterboxing and quiet intervals. There is no general shot detector in this release. Sparse speech timings and tight duration limits can leave no distinct legal cut for a platform.
- Synthetic metrics validate planted events. Real detection quality, music false positives, editorial acceptance, inference speed and financial outcomes remain unmeasured.

## Files changed against main

```text
.agents/skills/clipper-review/SKILL.md
.agents/skills/clipper-verify/SKILL.md
.github/workflows/test.yml
AGENTS.md
README.md
TASKS.md
clipper/campaign.py
clipper/cli.py
clipper/db.py
clipper/defaults.yaml
clipper/detect.py
clipper/evaluate.py
clipper/fixture.py
clipper/layout.py
clipper/media.py
clipper/pipeline.py
clipper/render.py
clipper/review_app.py
clipper/signals/visual.py
clipper/subtitles.py
clipper/variants.py
docs/PRODUCTION_HARDENING_REPORT.md
pyproject.toml
tests/test_campaign_policy.py
tests/test_layout.py
tests/test_production_hardening.py
tests/test_review_fixes.py
tests/test_subtitles_variants.py
```
