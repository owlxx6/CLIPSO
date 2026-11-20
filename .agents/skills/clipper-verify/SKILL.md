---
name: clipper-verify
description: Verify a CLIPSO change on the offline synthetic stream with pytest, environment diagnostics, a real vertical export, and frame inspection. Use for release checks or to verify a clipping change end to end.
---

# Verify a CLIPSO checkout

Work from the checkout being tested. Read `AGENTS.md` and record the branch, revision and any uncommitted changes. Use a Python environment that can import the project and run pytest. When FFmpeg is installed outside PATH, set `CLIPPER_FFMPEG_DIR` to the directory containing both executables.

1. Choose a fresh scratch directory. Point `CLIPPER_HOME` and `CLIPPER_DATA_DIR` there and clear unrelated `CLIPPER_CONFIG` overrides. A reused demo database can trigger the duplicate guard on a second run.
2. Run `python -m pytest -q -ra` and `python -m clipper doctor`. Report failures and skips separately, including optional Streamlit or face-fixture coverage. Record the FFmpeg build, encoder and Python version used.
3. Run `python -m clipper demo`. The built-in fixture plants three highlights. Check its reported precision@3 and recall and the output paths for the requested platforms. A successful process can still have blocked an export.
4. Probe an actual MP4 and inspect its sibling JSON, caption and cover. Compare encoded dimensions, streams and duration with the planned interval and render config. To inspect speech, pick a timed word inside the source interval, subtract the clip's start and extract a frame at that output time. Open the frame and describe the visible captions, hook if present, layout and campaign label when applicable. When testing configurable disclosure, use a synthetic campaign or a temporary label setting and identify which one you exercised.
5. Record the result in the requested handoff or project log. Include the frame path, the configurations exercised and any environmental limits. Use synthetic media throughout verification.
