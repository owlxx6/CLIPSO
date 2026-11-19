@AGENTS.md

## Notes for Claude Code

- Iterate with `pytest -q -m "not slow"`, run the full `pytest -q` before you say a task is done.
- Use plan mode before touching `clipper/dedup.py`, the database schema or `clipper/variants.py`: those carry
  the rules above.
- After a change to layouts, captions or encoders, render one clip with `clipper demo` and look at a frame.
- Keep TASKS.md current: it is how Codex knows what you did.
