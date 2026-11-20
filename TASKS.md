# TASKS

Shared log for the humans and agents working on this repo (see AGENTS.md). One task per agent at a time.

## In progress

- **2026-11-20 · Ready for review on `fix/review-2026-11-20`.** Completed issues 1–6. Rendering reuses same-source speech after a Whisper setting change and checks caption coverage. Legacy speech is adopted and can be regenerated. Migration distinguishes reproducible auto hooks from manual or uncertain hooks. Replacement tolerates locked old files and reports them. OpenCV still images support Unicode paths. Paid exports use the configured disclosure with a default for empty text.
- Added the repository skills `clipper-verify` and `clipper-review` and validated both skill definitions.
- **Verification.** Windows, Python 3.13.17, FFmpeg 4.4.1 and libx264. All 78 tests passed with zero skips. Doctor reported ready. The synthetic demo reached precision@3 and recall of 1.0 and produced six 1080×1920 exports with audio, covers and sidecars. A separate CLI run adopted a first-version transcript and rendered a synthetic campaign with `--no-disclosure`. The inspected frame showed the configured `PARTENARIAT RÉMUNÉRÉ` label, hook and highlighted word. Speech verification used fixture transcripts and a controlled transcriber.

### Open findings from the 20 November review

| Priority | Reproduced behavior | Acceptance test for a follow-up |
|---|---|---|
| P2 | [CLI labeling](clipper/cli.py#L224) accepts a candidate from source A when the command specifies source B. It returned success and wrote the label to B, which can distort evaluation and tuning. | Mismatched source and candidate IDs return an error and neither source gains a label. A matching pair still succeeds. |
| P2 | [Ingestion](clipper/pipeline.py#L51) checks that sidecars exist, then commits the source before copying them. A simulated copy permission error on an existing chat file left one source row. | Inject a sidecar copy failure and verify that no source or partial sidecar is visible. Retrying creates one complete source. |

## Next, in order

1. **Calibrate on real VODs.** One GTA VI VOD per target streamer with chat and viewer clips. Label 20 to 30
   moments in the review app, `clipper tune <id> --write`, re-analyze, check `clipper eval` on a second VOD
   of the same streamer. Write the precision@10 before and after here. Until this is done the default
   weights are educated guesses.
2. **Music and radio.** GTA radio and stream music make loud stretches that are not reactions. Detect music
   (steady beat, spectral flatness) and lower the audio score there. Warn when a clip carries licensed music:
   platforms mute or claim it.
3. **Fetch helpers.** `clipper fetch-chat <vod>` (wraps TwitchDownloaderCLI) and `clipper fetch-clips <vod>`
   (Helix Get Clips, keeps only clips whose `video_id` is the VOD). Credentials from environment variables.
4. **Hook ideas from a local model** (Ollama): three French hooks per moment from the transcript, 8 words at
   most, true to what happens. The user picks one in the review app.
5. **Publishing through official APIs.** YouTube Data API upload with scheduled `publishAt`; Instagram Graph
   API Reels publishing; TikTok stays manual in TikTok Studio until an app passes the Content Posting API
   audit. Always an explicit command, never automatic. Record the post with `db.add_post`.
6. **Post tracking.** Views and earnings per post (the `posts` table already has the columns): manual entry
   first, then YouTube Analytics and Instagram Insights.
7. **Facecam edge cases.** Streamer away from the camera, green-screen cutouts with no rectangle, two cams.
8. **Speed.** Hardware decoding for analysis, renders in parallel, a render queue in the review app.
9. **Catch typos in settings.** Unknown keys in config files are ignored silently today; warn in `doctor`.

## Done

- 2026-11-19: first working version.
  - Ingest with rights; signals: audio excitement, speech rate, French keywords and laughter, chat speed and
    emotes, Twitch viewer clips, cutscene bars.
  - Detection with cuts snapped to speech pauses; 15 to 34 s (campaign) or 30 to 60 s (story) clips.
  - Layouts: facecam on top (YuNet face detection grown to the overlay border; the gameplay crop steps
    aside so the cam is not shown twice), blurred background, centre
    crop; layout preview PNG.
  - Captions word by word with highlight, hook box, PUBLICITÉ tag below the app top bar; Anton font;
    loudness -14 LUFS; covers, captions `.txt`, metadata `.json`.
  - One version per platform; duplicate guard before rendering (same time, or same footage from another copy
    of the VOD); `clipper posted` protects live exports.
  - Review app; labels; precision/recall@k; weight tuning saved per streamer.
  - Deterministic synthetic stream; 40 tests in about 2 minutes.

- Stabilization for initial CLIPSO publication: rights and campaign gates, validated staged exports, stable candidate IDs and hooks, one-to-one evaluation, keyed caches, SQLite migration and Windows CLI output. Reconstructed from the first-version archive and checked against independent review probes.
