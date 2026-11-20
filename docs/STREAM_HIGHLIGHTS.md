# Stream Highlights V1

Stream Highlights ranks local GTA recordings for a human editor. It can return an empty list. A candidate stores its component scores, evidence times, estimated setup, action, peak, reaction end when observed, proposed end, cut warning and up to three editable hooks. The classic detector remains the default.

## Try it on Windows

From this checkout in PowerShell, with FFmpeg and ffprobe on PATH, run the following. Alternatively set `$env:CLIPPER_FFMPEG_DIR` to the directory containing both executables.

```powershell
py -3.11 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -e ".[review,dev]"
& $py -m clipper doctor
$video = Read-Host "Path to your local 15-minute recording"
$rights = Read-Host "Permission reference and permitted use"
& $py -m clipper ingest "$video" --rights "$rights"
$id = [int](Read-Host "Source ID printed by ingest")
& $py -m clipper analyze $id --mode classic --no-transcribe
& $py -m clipper analyze $id --mode stream_highlights --no-transcribe
& $py -m clipper candidates $id --mode stream_highlights --top 10
& $py -m clipper review
```

Keep each application's writable database with its own checkout when comparing V1 with an older release. For a source already registered in this checkout, use its existing ID. To inspect another checkout's data in isolation, copy the database using SQLite backup, the source sidecars and relevant configuration to a separate CLIPPER_HOME. For timed speech, use `ingest --transcript FILE`, or bind a verified transcript with `clipper import-transcript $id FILE`. With faster-whisper and a local model installed, omit `--no-transcribe` to generate words. The review app's Analyze button reuses available speech and works offline.

Automatic facecam detection uses the bundled YuNet detector (or the existing OpenCV fallback). If necessary, inspect the frame and enter the actual normalized overlay rectangle for that recording.

```powershell
& $py -m clipper layout $id --detect --preview .\layout-check.png
$box = Read-Host "Observed facecam bounds as x,y,width,height in fractions"
& $py -m clipper layout $id --set "$box" --kind stacked
& $py -m clipper analyze $id --mode stream_highlights --no-transcribe
```

In **Stream Highlights**, inspect **Score and timed evidence**, choose a framing, and enable the preview. Save the desired in/out times and hook, including an empty hook. The vertical preview uses the same base framing filter as export. Captions, hook, disclosure, encoder and platform zoom are applied in the final render. Good/Bad/Clear remain available. A saved interval must include the measured peak. To render a local review file, choose a candidate ID from the Stream Highlights list.

```powershell
$candidate = [int](Read-Host "Reviewed candidate ID")
& $py -m clipper render $id --ids $candidate --platforms tiktok --layout stacked
& $py -m clipper exports --source $id
```

Choose `--layout center_crop` for a classic vertical crop or `blur_fit` to keep the full frame. An explicit stacked layout needs a camera rectangle. A platform variant keeps the entire reviewed interval and may expand into nearby context for clean, distinct cuts. If campaign length or duplicate constraints conflict with a proposed interval, review the bounds or select fewer platforms. Existing rights, disclosure, posted-export and validation checks apply.

## Measurements and interpretation

| Channel | What the V1 actually measures |
|---|---|
| Audio | A rapid dBFS rise against the previous hop and a local median, changes in estimated pitch across periodic windows, time-aligned speech density, exclamations and transcribed laughter markers. A steady loud track contributes little. Mixed audio can include music, game dialogue or another speaker. |
| Facecam | Face detection within the configured or detected overlay, association by proximity on successive samples, displacement and intensity change on the detected face. Size and sharpness attenuate confidence. A missing or unresolved face contributes zero. |
| Gameplay | The fraction of changed pixels outside the camera overlay, its deviation from local activity and the distance between successive grayscale histograms. Cuts, camera motion, menus and flashes can all produce strong responses. |
| Synchronization | A visual event paired with audio or tracked-face activity in a configurable following window. The report stores both timestamps, response channel and strength. |
| Narrative and context | An adjacent question and complete response, transcribed utterance limits, speech pauses, shot-change cues and settling time. A bounded gap in unfinished dialogue can be bridged when activity continues without a detected shot change. |
| Hooks | Complete, short questions or imperatives with recognizable French or English openings, within six seconds of the measured peak. Multi-sentence fragments are excluded. Incomplete or weak evidence can leave all three slots empty. |

Scores use the weights in `stream_highlights.weights`. They combine audio reaction, facecam activity, gameplay activity, event/reaction synchronization, narrative coherence, novelty and context completeness. The review view and export metadata retain each weighted contribution from the analysis run. Their sum recovers the reported total to the displayed precision. The reported confidence is a qualitative heuristic. A chaos label requires both a spoken event term and synchronized observations. The underlying event and the speaker's emotion still require review. Colour-only changes, occlusion, game voices and inaccurate ASR can all affect the result.

The detector searches around observed activity and protects complete spoken words where timestamps allow. The configurable duration cap, an unfinished utterance, sustained activity, a bridged dialogue gap or a recording edge can trigger a cut warning. A pause can belong to the same scene. Review surrounding footage before accepting the conclusion.

The YAML keys and defaults are in [defaults.yaml](../clipper/defaults.yaml). Use `analysis.mode: stream_highlights` to make this ranking the local default. Adjust `stream_highlights.min_s`, `max_s`, `speech_pause_s`, `dialogue_bridge_s`, `sync_s` and `min_score` in a local or streamer config. Expensive stream measurements are cached using the recording digest, sampling parameters and camera ROI. Changing scoring weights reuses the measurements. The two rankings share time-based labels and retain their own candidates and edited hooks. V1 uses the existing SQLite schema. Classic `tune` continues to tune the classic detector.

## Before and after on the 15-minute recording

The comparison used a private SQLite backup, the same source, its existing timed transcript, the four existing labels and the local streamer settings. The clip was marked for local testing only. No private media, transcript, database or generated export is included in this branch.

| Mode | Candidates | First three peak times | Top candidate interval | Wall time | Sampled peak working set |
|---|---:|---|---|---:|---:|
| Classic, campaign target | 18 | 06:48.2, 09:32.8, 00:07.2 | 06:44.4–07:18.2 | 0.44 s | 45 MiB |
| Classic, story target | 15 | 06:48.2, 09:32.8, 00:07.2 | 06:13.9–07:23.0 | 0.36 s | 45 MiB |
| Stream Highlights, first activity scan | 29 | 03:16.2, 01:35.2, 06:12.2 | 03:13.0–03:35.2 | 5.34 s | 93 MiB |
| Stream Highlights, cached activity | 29 | Same order | Same interval | 0.39 s | 45 MiB |

The initial campaign interval reported by the tester was 06:34–06:57. The active local settings reproduced the reported story interval and produced the campaign interval in the table. At rank 11, Stream Highlights retained **06:34.81–07:39.03** around the dialogue, including 16.01 seconds after the previous story ending. It flagged the uncertain conclusion. Its first-ranked candidate had a gameplay change at 03:16.0 and possible mixed-audio response at 03:16.5. Twenty-seven of 29 candidates had no automatic hook. The top candidate had a punctuated but disfluent ASR passage, so the hook was left empty.

The two marked positive windows appeared at ranks 2 and 4 in both classic runs and ranks 6 and 10 in Stream Highlights. Two other windows were labeled negative. These sparse annotations do not establish precision over all proposals. A human editor still needs to judge whether the longer dialogue cut resolves the scene and whether the new top moments merit export.

All runs used cached common loudness/letterbox measurements, an existing ROI and transcript. Sampled cumulative CPU time for the process tree was approximately 0.23 s, 0.25 s, 49.58 s and 0.27 s respectively, measured at 100 ms intervals. CPU time can exceed wall time on a multicore machine. These single-workstation observations exclude ASR and initial camera discovery. For a repeatable comparison, use the same local configuration and source for both modes, record cold and warm runs separately, and freeze your event annotations before looking at either ranking. Time the commands with `Measure-Command`. Record missed events, incorrect speaker attribution, reactions cut short, excess lead-in and review time per accepted clip.

## Source inspection and V1 scope

| Project and revision | Implementation, dependencies and license inspected | Decision |
|---|---|---|
| [WowClip tracking](https://github.com/partyfly/wowclip/blob/c60753daded5543ea9e81375f04a543aef550945/scripts/wowclip-track-faces.py) and [timeline](https://github.com/partyfly/wowclip/blob/c60753daded5543ea9e81375f04a543aef550945/scripts/wowclip-build-timeline.py) | MIT. Python scripts associate detections and compile planned ranges. The manifest includes OpenCV, faster-whisper, Pillow, NumPy and yt-dlp. | Keep ROI-limited geometric tracking and explicit source times. V1 uses CLIPSO's existing layout and renderer. |
| [Auto-editor motion](https://github.com/WyattBlue/auto-editor/blob/b944b7e920f658688d411b2e1f8d8fbe3fd537a2/src/analyze/motion.nim) and [audio](https://github.com/WyattBlue/auto-editor/blob/b944b7e920f658688d411b2e1f8d8fbe3fd537a2/src/analyze/audio.nim) | Unlicense. Nim and FFmpeg-based amplitude and changed-pixel analysis. Build metadata includes Nim and nimcrypto. | Take low-cost activity measurements and keep silent gameplay inside selected intervals. |
| [Kinocut receipt](https://github.com/KyaniteLabs/kinocut/blob/d46975a39c68bdd163f93d97faf37eb1941c51c2/kinocut/workflow/receipt.py) | Apache-2.0. Python workflow receipt assembly and atomic persistence. The project includes an MCP dependency and FFmpeg tooling. | Attach the proposal evidence to the already validated CLIPSO export sidecar. |
| [AI clipping skill](https://github.com/social-media-skills/ai-clipping-skill/tree/1ad8906ca39f1a30d650f3e9f1e38baa99195c27/skills/ai-clipping-skill) | MIT. Skill, scripts, examples and campaign checklists describe vertical composition, approval and proof. External publishing integrations are separate. | Retain human review, local export and existing campaign checks. |

No upstream code or new runtime dependency was added. The V1 runs on CPU with the existing FFmpeg, NumPy and OpenCV installation. Semantic scene recognition and a local LLM are deferred. A future local vision model should identify its weights and license, run on a measured Windows CPU budget and outperform these cues on independently annotated footage before its labels enter hooks or ranking.
