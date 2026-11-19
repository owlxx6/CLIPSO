# GTA AI Clipper

Finds the best moments in a GTA VI stream VOD and renders them as vertical clips (1080x1920) for TikTok,
Instagram Reels and YouTube Shorts: facecam on top, gameplay below, word-by-word captions, a hook line and,
for paid campaigns, a "PUBLICITÉ" tag. You review the picks, mark good and bad ones, and the tool tunes itself
to what you would have clipped.

Everything runs on your computer. Nothing is posted for you.

```
VOD + chat + Twitch clips ──> signals ──> ranked moments ──> you review / label ──> tune weights
                                                                      │
                                                                      └──> render 1 version per platform ──> you post
```

**Signals** (one value every 0.5 s of the VOD):

| signal | what it catches | needs |
|---|---|---|
| `audio` | the streamer gets loud compared with the minute around | nothing |
| `speech_rate` | they suddenly talk fast | transcript |
| `keywords` | "oh mon dieu", "je suis mort", laughter (mdr, ptdr, haha) | transcript |
| `chat` | chat speeds up, KEKW / LUL / 💀 spam | chat log |
| `viewer_clips` | viewers made Twitch clips there | Twitch clip list |

Chat and viewer clips are the strongest predictors: give them whenever you can.

## Ground rules (built in)

- **Rights first.** `ingest` refuses a video without `--rights` (the campaign or the streamer's written OK).
  The rights note is copied into every export's `.json`.
- **You post, by hand or through official APIs.** No posting bots, no browser automation, one account per
  platform.
- **Paid clips are labelled.** A source with `--campaign` or a `campaign:` / `content-rewards:` rights note always burns in "PUBLICITÉ" (French law on influencer marketing).
  Use `--disclosure` to label other sources. Also switch on the platform's own label when you post: *content disclosure* on TikTok, *paid partnership* on
  Instagram, *includes paid promotion* on YouTube.
- **One version per platform, never twice.** A moment already exported for a platform is skipped, even from
  another copy of the VOD (see [Duplicate guard](#duplicate-guard)). The three platform versions differ by
  cut points, zoom and caption colour: real edits, not tricks to fool duplicate detection (no noise, no
  mirroring, no pitch shifting).
- **No story spoilers by accident.** Moments inside a cutscene (black bars) are flagged `[check spoiler]`.
- **No gambling promos.** Don't take campaigns for casinos or betting sites.

## Install

### Windows 11

```powershell
winget install Python.Python.3.11
winget install Gyan.FFmpeg
# open a NEW terminal so ffmpeg is on PATH, then in the repo folder:
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[transcribe,review,dev]"
clipper doctor
```

### macOS

```bash
brew install python@3.11 ffmpeg-full
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[transcribe,review,dev]"
clipper doctor
```

Use `ffmpeg-full`, not `ffmpeg`: Homebrew's plain `ffmpeg` has no libass since January 2026, so it cannot
draw captions. The tool finds `ffmpeg-full` by itself even though Homebrew keeps it off PATH.

`clipper doctor` must end with `ready`. Lines marked `--` are optional (a hardware encoder you don't have).

### Faster renders (optional)

Copy `config/local.example.yaml` to `config/local.yaml` and set the encoder:

| machine | `render.encoder` |
|---|---|
| Intel Arc GPU or Intel iGPU (Quick Sync) | `h264_qsv` |
| Mac | `h264_videotoolbox` |
| anything | `libx264` (default) |

## Try it: the demo

```bash
clipper demo
```

Generates a 3-minute fake stream with three known highlights (loud audio, chat bursts, a viewer clip, a
cutscene), finds them, prints precision and recall, and renders the top 2 moments for the 3 platforms. Takes
1 to 2 minutes. Then:

```bash
clipper review
```

opens the review app in the browser (http://localhost:8501).

## Workflow on a real VOD

### 1. Get the VOD, the chat and the viewer clips

Only VODs you have the right to clip: a campaign on Content Rewards / Whop that allows it, or the streamer's
written OK.

**VOD and chat** with [TwitchDownloader](https://github.com/lay295/TwitchDownloader) (GUI or CLI):

```powershell
TwitchDownloaderCLI.exe videodownload --id 2612345678 -o D:\vods\2612345678.mp4
TwitchDownloaderCLI.exe chatdownload --id 2612345678 -o D:\vods\2612345678_chat.json
```

The chat can also be a JSON list (`[{"t": 12.3, "text": "KEKW"}]`) or a CSV with `time,message` columns.

**Viewer clips** come from the Twitch API (`Get Clips`, field `vod_offset`). You need a free app at
https://dev.twitch.tv/console for a client id and secret. In PowerShell:

```powershell
$cid = "your-client-id"; $secret = "your-client-secret"; $login = "streamer_login"; $vod = "2612345678"
$tok = (Invoke-RestMethod -Method Post "https://id.twitch.tv/oauth2/token?client_id=$cid&client_secret=$secret&grant_type=client_credentials").access_token
$h = @{ "Client-Id" = $cid; "Authorization" = "Bearer $tok" }
$uid = (Invoke-RestMethod -Headers $h "https://api.twitch.tv/helix/users?login=$login").data[0].id
# clips created between the stream start and a day later (UTC), at most 100 per call
$res = Invoke-RestMethod -Headers $h "https://api.twitch.tv/helix/clips?broadcaster_id=$uid&started_at=2026-11-18T18:00:00Z&ended_at=2026-11-19T18:00:00Z&first=100"
@{ data = @($res.data | Where-Object { $_.video_id -eq $vod }) } | ConvertTo-Json -Depth 5 | Out-File "D:\vods\$($vod)_clips.json" -Encoding utf8
```

Keep only the clips of this VOD (`video_id`), as above: offsets from another VOD would point at the wrong
moments.

### 2. Streamer profile (once per streamer)

Copy `config/streamers/example.yaml` to `config/streamers/<name>.yaml` and edit it: display name, their
catchphrases, their channel emotes, the facecam box if detection gets it wrong, the disclosure tag for paid
campaigns.

### 3. Ingest and analyze

```powershell
clipper ingest D:\vods\2612345678.mp4 --rights "content-rewards:<campaign>" --campaign "<campaign>" `
    --streamer <name> --language fr --chat D:\vods\2612345678_chat.json --clips D:\vods\2612345678_clips.json
clipper analyze 1
clipper candidates 1
```

The first `analyze` transcribes the VOD with faster-whisper (the model downloads once). It is the slow step,
roughly 10 minutes per hour of VOD with the default `small` model on a recent CPU, and runs once per VOD: the
transcript and the loudness scan are cached, so later runs take seconds. `--no-transcribe` skips it (the
speech signals are then off).

### 4. Check the layout

```bash
clipper layout 1 --preview layout.png --at 600
```

Open `layout.png`: red is the facecam box, green the gameplay crop. If the red box is wrong:

```bash
clipper layout 1 --set 0.72,0.04,0.26,0.30      # x, y, width, height as fractions of the frame
```

or put `facecam:` in the streamer profile so every VOD of that streamer uses it. No facecam? The clip uses
the blurred-background layout (`--kind blur_fit`).

### 5. Review and label

```bash
clipper review
```

For each moment: watch the preview, read the transcript, fix the hook, click **Good** or **Bad**, render.
Labels are what the tuning learns from: 20 to 30 per streamer is a good start, bad ones count as much as good
ones.

You can also write down the moments you would clip while watching the VOD (`start,end` in seconds or
`HH:MM:SS`) and load them:

```bash
clipper label 1 --truth moments.csv
```

### 6. Measure, then tune

```bash
clipper eval 1                  # how many of the top picks are good (precision) and how many good moments were found (recall)
clipper tune 1 --write          # search the signal weights that match your labels best -> config/tuned/<name>.yaml
clipper analyze 1               # re-rank with the new weights
clipper eval 1
```

`tune` never writes weights that score worse than the current ones. Tune on one VOD and check `eval` on
another VOD of the same streamer: weights that only fit one VOD are no use.

### 7. Render

```bash
clipper render 1 --good-only --platforms tiktok,instagram,youtube --disclosure
clipper render 1 --ids 12 --hook "Il ne s'attendait pas à ça" --replace     # new hook for a clip not posted yet
```

Each clip lands in `data/exports/source_<id>/candidate_<id>/`:

| file | what |
|---|---|
| `tiktok_v1.mp4` | the clip, 1080x1920, 30 fps, -14 LUFS |
| `tiktok_v1.jpg` | cover frame |
| `tiktok_v1.txt` | caption to paste |
| `tiktok_v1.json` | rights, campaign, source timestamps, hook, layout: keep it for campaign proof |

### 8. Post, then record the link

By hand for now: TikTok Studio on the web (upload and schedule), Instagram (app or Meta Business Suite),
YouTube Studio. Switch on the platform's paid-content label for campaign clips, then submit the post link on
the campaign page.

Then tell the tool it is live, in the review app (**Mark as posted** under the export) or here:

```bash
clipper exports --platform tiktok                                   # export ids and what is already posted
clipper posted 7 --url https://www.tiktok.com/@you/video/123456      # export #7 is live
```

A posted export is never replaced or deleted, and its moment is never rendered again for that platform.

## Tuning cheat sheet

Put these in `config/streamers/<name>.yaml` (one streamer) or `config/local.yaml` (everything). Run
`clipper analyze <id>` again after a change: it reuses the cached transcript and scans.

| problem | setting |
|---|---|
| good moments are missed, bad ones picked | label more, then `clipper tune` (weights in `detection.weights`) |
| loud but boring moments (music, sirens, shooting) | lower `detection.weights.audio`, raise `keywords` / `chat` |
| the setup before the moment is cut | raise `detection.pre_roll_s` (default 14) |
| the reaction is cut | raise `detection.post_roll_s` (default 8) |
| cuts in the middle of a sentence | raise `detection.snap_search_s` (default 3) |
| two clips of the same scene | raise `detection.min_gap_s` (default 30) |
| chat peaks land after the moment | raise `signals.chat.delay_s` (default 6) |
| the streamer's own catchphrases | `signals.keywords.fr` (copy the default list, then add) |
| the channel's emotes | `signals.chat.emotes` (copy the default list, then add) |
| the campaign wants 30 to 60 s | `platforms.<platform>.target: story` |
| captions too big or too small | `render.subtitles.font_size` (96), `render.hook.font_size` (72) |
| Whisper too slow | `analysis.transcript.model: base` |

All settings and their defaults are in `clipper/defaults.yaml`. Lists replace the default list instead of
adding to it.

## Duplicate guard

Before rendering, for each platform:

1. **Same VOD, same time** (at least 50% overlap with an earlier export): skipped.
2. **Same footage from another copy of the VOD**: one 256-bit picture hash per second of gameplay (facecam
   left out), compared with every earlier export of that platform. Similar enough, and it is skipped.

Captions, hook, colours and zoom are not part of the comparison, so a new hook does not make an old moment
new. To redo an export that was **not posted yet** (new hook, other layout), use `--replace` or the
review app's Render button. Exports recorded with `clipper posted` are never replaced.

## Troubleshooting

| message | fix |
|---|---|
| `ffmpeg not found` | install it (see Install) and open a new terminal, or set `CLIPPER_FFMPEG_DIR` to the folder holding `ffmpeg` and `ffprobe` |
| doctor: `filter subtitles (libass)` missing | your FFmpeg has no libass. Windows: `winget install Gyan.FFmpeg`. Mac: `brew install ffmpeg-full` |
| `h264_qsv` render fails | update the Intel Arc driver, or go back to `render.encoder: libx264` |
| `faster-whisper is not installed` | `pip install -e ".[transcribe]"`, or analyze with `--no-transcribe` |
| the review app does not open | go to http://localhost:8501 |
| a path with spaces fails | put it in quotes |

## Development

```bash
pytest -q                 # everything (about 2 minutes: it builds small test videos with FFmpeg)
pytest -q -m "not slow"   # the fast part
```

Coding agents: read [AGENTS.md](AGENTS.md). Work in progress and next steps: [TASKS.md](TASKS.md).

## Credits

- Caption font: Anton by Vernon Adams, SIL Open Font License 1.1 (`clipper/assets/fonts/OFL.txt`).
- Face detector: YuNet from the OpenCV Zoo, MIT licence (`clipper/assets/models/YUNET_LICENSE.txt`).

## Upgrading the first-version archive

The first database connection migrates the original SQLite schema transactionally. Older exports and posts keep their IDs. Reanalysis retains edited hooks and historical candidates used in exports. Review the existing rights notes before rendering.

New imports keep their supplied transcript separately from generated Whisper output. Generated transcripts depend on the source file and Whisper settings. The first version used the same filename for both. To keep a verified legacy transcript, copy `data/sources/<id>/transcript.json` to `transcript.imported.json` in that directory. Otherwise run analysis with Whisper enabled. A forced run rebuilds generated signals and preserves supplied transcripts.
