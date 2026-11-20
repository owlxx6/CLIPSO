---
name: clipper-review
description: Review a CLIPSO branch or working-tree diff for regressions in transcript reuse, candidate history, paid disclosure, export replacement, media handling and duplicate prevention. Use for a code review or an audit of clipping behavior.
---

# Review a CLIPSO change

Establish the comparison branch and inspect both its diff and pending local work. Read `AGENTS.md`, the affected entry points and their tests. Trace a behavior through both the CLI and the Streamlit app when they share the pipeline.

- **Speech and hooks.** Distinguish imported, generated and legacy transcripts. Try changed Whisper settings with the same media, a changed source, disabled transcription and a forced rerun. Check the actual caption events or a rendered frame. Examine how manual edits, cleared hooks, and unknown legacy hook origins survive reanalysis.
- **Rights and exports.** Follow the effective rights and disclosure decisions through subtitles and JSON. For replacement, follow the old file, sidecars, database row and posted guard across encoder failure, publication and cleanup. Check per-platform duplicate evidence and how decode errors surface.
- **Windows media paths.** Exercise the touched path with spaces, accents and non-Latin characters. For still images, decode the file and verify its dimensions. For CLI output, check the exit code under the relevant console encoding.
- **Regression evidence.** Reuse the synthetic fixtures in `tests/conftest.py`. Fault-inject only in a test-owned folder or database. A test that stops at ASS generation proves the events were prepared. A real encode and an opened frame can verify that the label is visible.

For each actionable finding, give severity, file and line, triggering conditions, observed effect, and a concise acceptance test. Distinguish reproduced failures from an inference. When asked for review only, present findings. When fixes are requested, verify each against its own regression and finish with the requested end-to-end checks.
