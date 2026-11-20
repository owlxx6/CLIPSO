# TASKS

Shared log for the humans and agents working on this repo (see AGENTS.md). One task per agent at a time.

## Production hardening review

- Feature branch `fix/production-hardening` includes the reviewed fixes, the repository verification/review skills and the remaining November 20 audit work. Cross-source labeling and sidecar-copy rollback now have regression tests.
- Imported transcript reuse requires explicit recording identity. Global highlight evaluation, deterministic speech-aware platform cuts, streamed FFmpeg failure detection and campaign evidence checks are implemented.
- All 128 tests passed locally on Windows with zero skips. Track the Python 3.11 Linux/Windows checks and the six-export demo in the [hardening report](docs/PRODUCTION_HARDENING_REPORT.md). The first authorized real-recording benchmark is pending access to recordings, complete event labels and verified campaign terms.
- Phase 3 begins with a single-cut EditPlan round trip through the existing renderer and an ExportReceipt replay check. Evaluate pacing and facecam experiments separately after measuring the current editor on the held-out recordings.

## November 20 review history

The earlier `fix/review-2026-11-20` branch passed 78 synthetic tests, including the original stabilization probes. It preserved edited hooks, made Unicode image paths work, used the configured paid label and handled locked replacement files. This sprint supersedes automatic legacy transcript adoption with explicit operator binding.

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
