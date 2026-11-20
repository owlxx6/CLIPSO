"""Review app (Streamlit): preview candidates, label them good or bad, render the good ones.

Start it with `clipper review`. Labels are what `clipper eval` and `clipper tune` learn from.
"""

from __future__ import annotations

import streamlit as st

from clipper import config, db, evaluate, media, opening, pipeline, stream_highlights, layout

LABEL_NAMES = {1: "good", 0: "bad", None: "not reviewed"}

st.set_page_config(page_title="GTA AI Clipper - review", layout="wide")


conn = db.connect(config.load_config())
st.title("GTA AI Clipper - review")
sources = db.list_sources(conn)
if not sources:
    st.info("No video yet. Run `clipper demo`, or `clipper ingest VIDEO --rights ...` then `clipper analyze ID`.")
    st.stop()

src = st.sidebar.selectbox(
    "Video", sources, format_func=lambda s: f"#{s['id']} - {s['title']} - {media.fmt_ts(s['duration'] or 0)}")
cfg = pipeline.source_config(src)
modes = ["classic", stream_highlights.MODE]
mode = st.sidebar.selectbox("Analysis mode", modes, index=modes.index(cfg["analysis"]["mode"]),
                            format_func=lambda value: "Stream Highlights" if value == stream_highlights.MODE else "Classic")
if st.sidebar.button("Analyze recording", key="analyze_source"):
    try:
        with st.spinner("Analyzing local recording ..."):
            pipeline.analyze(conn, cfg, src["id"], mode=mode, transcribe=False, log=st.caption)
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(str(exc))
cands = db.list_candidates(conn, src["id"], mode)
labels = db.list_labels(conn, src["id"])
show = st.sidebar.radio("Show", ["all", "not reviewed", "good", "bad"])
platforms = st.sidebar.multiselect("Platforms to render", list(cfg["platforms"]), default=list(cfg["platforms"]))
st.sidebar.caption(f"Rights: {src['rights']}")

if not cands:
    st.warning("No candidates in this mode. Analyze this recording or change the mode.")
    st.stop()

top_n = len(cands) if len(cands) <= 5 else st.sidebar.slider("Candidates shown", 5, len(cands), min(20, len(cands)))
reviewed = [db.label_for(labels, c["start_s"], c["end_s"]) for c in cands]
n_reviewed = sum(x is not None for x in reviewed)
st.sidebar.metric("Good / reviewed", f"{sum(x == 1 for x in reviewed)} / {n_reviewed}")
truth = evaluate.truth_from_labels(labels)
if truth:
    m = evaluate.evaluate([{"start": c["start_s"], "end": c["end_s"], "peak": c["peak_s"]} for c in cands], truth,
                          k=min(10, len(cands)))
    st.sidebar.caption(f"precision@{m['k']} {m['precision']} - recall {m['recall']} (vs your good labels)")

for c in cands[:top_n]:
    lab = db.label_for(labels, c["start_s"], c["end_s"])
    if (show == "not reviewed" and lab is not None) or (show == "good" and lab != 1) or (show == "bad" and lab != 0):
        continue
    with st.container(border=True):
        left, right = st.columns([3, 2])
        with left:
            flags = c.get("flags") or {}
            start, end = flags.get("review_bounds", [c["start_s"], c["end_s"]])
            saved_start, saved_end = start, end
            layout_kind = None
            if mode == stream_highlights.MODE:
                st.write(f"{flags['moment_type']} · {flags['confidence']} confidence")
                st.write(flags["reason"])
                if "payoff_status" in flags:
                    st.caption(f"Payoff evidence: {flags['payoff_status']} · camera type: {flags['camera_type']}")
                st.caption(" · ".join(f"{name.replace('_', ' ')} {media.fmt_ts(t) if t is not None else 'unobserved'}" for name,t in flags["timing"].items()))
                with st.expander("Score and timed evidence"):
                    quality_parts = {k: v for k, v in flags.get("contributions", {}).items() if k != "opening_bonus"}
                    st.json({"measurements": c["signals"], "highlight_quality_components": quality_parts,
                             "observations": flags["evidence"]})
                if flags["cut_risk"]:
                    st.warning("The ending or setup is uncertain. Check the surrounding footage.")
                start = st.number_input("Start (seconds)", min_value=0., max_value=float(src["duration"]),
                                        value=float(start), step=0.25, key=f"start_{c['id']}")
                end = st.number_input("End (seconds)", min_value=0., max_value=float(src["duration"]),
                                      value=float(end), step=0.25, key=f"end_{c['id']}")
                if "opening_context" in flags and start < end:
                    current = opening.scored_flags(flags, start, end)
                    st.write(f"Highlight quality {current['highlight_quality']:.2f} · "
                             f"Opening strength {current['hook_strength_score']:.2f}/1 · "
                             f"Ranking bonus +{current['opening_bonus']:.2f}")
                    if current["weak_opening"]:
                        st.warning(current["opening_note"] or "Weak opening. Review the first seconds of this cut.")
                    if current["opening_bonus_note"]:
                        st.caption(current["opening_bonus_note"])
                    with st.expander("First-seconds evidence"):
                        left_s, right_s = current["opening_window"]
                        st.caption(f"Source interval {left_s:.2f}–{right_s:.2f} s")
                        for cue in current["opening_evidence"]:
                            detail = f"“{cue['text']}”" if "text" in cue else ""
                            if "support_kind" in cue:
                                detail += f" · {cue['support_kind']} by {cue['support']:.2f} s in source"
                            st.caption(f"+{cue['clip_offset']:.2f} s · {cue['kind']} · {cue['score']:.2f} {detail}")
                elif "opening_context" not in flags:
                    st.caption("Reanalyze this recording to measure opening strength.")
                layout_kind = st.selectbox("Framing", ["auto", "stacked", "center_crop", "blur_fit"],
                                           key=f"layout_{c['id']}")
                if layout_kind == "stacked" and not layout.resolve(
                        cfg, pipeline.workdir(cfg, src["id"]), "stacked")["facecam"]:
                    st.info("A facecam box is required for stacked framing. Preview uses the blurred background "
                            "until you set the overlay with clipper layout.")
                for suggestion in flags["hooks"]:
                    if st.button(suggestion, key=f"quote_{c['id']}_{suggestion}"):
                        st.session_state[f"hook_{c['id']}"] = suggestion
            st.markdown(f"**#{c['id']} - rank {c['rank']}** - {media.fmt_ts(saved_start)} -> "
                        f"{media.fmt_ts(saved_end)} - {saved_end - saved_start:.0f}s - score {c['score']:.2f} - "
                        f"*{LABEL_NAMES[lab]}*")
            sig = c.get("signals") or {}
            st.caption(" - ".join(f"{k} {v:.1f}" for k, v in sorted(sig.items(), key=lambda kv: -kv[1]) if v > 0)
                       or "no signal detail")
            if (c.get("flags") or {}).get("spoiler_review"):
                st.warning("Looks like a cutscene: check for story spoilers before posting.")
            if c.get("text"):
                st.write(f"\"{c['text'][:300]}\"")
            hook = st.text_input("Hook", value=c.get("hook") or "", key=f"hook_{c['id']}")
            if mode == stream_highlights.MODE and st.button("Save bounds and hook", key=f"save_{c['id']}"):
                try:
                    db.save_stream_review(conn, c["id"], start, end, hook)
                    st.toast("Review saved")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
            b1, b2, b3, b4 = st.columns(4)
            if b1.button("Good", key=f"good_{c['id']}"):
                db.add_label(conn, src["id"], c["start_s"], c["end_s"], 1)
                st.rerun()
            if b2.button("Bad", key=f"bad_{c['id']}"):
                db.add_label(conn, src["id"], c["start_s"], c["end_s"], 0)
                st.rerun()
            if b3.button("Clear", key=f"clear_{c['id']}"):
                db.clear_label(conn, src["id"], c["start_s"], c["end_s"])
                st.rerun()
            if b4.button("Render", key=f"render_{c['id']}", type="primary", disabled=not platforms):
                if mode != stream_highlights.MODE and hook != (c.get("hook") or ""):
                    db.set_hook(conn, c["id"], hook)
                try:
                    old_bounds = flags.get("review_bounds", [c["start_s"], c["end_s"]])
                    if mode == stream_highlights.MODE and ([start, end] != old_bounds or hook != (c.get("hook") or "")):
                        db.save_stream_review(conn, c["id"], start, end, hook)
                    with st.spinner("Rendering ..."):
                        # replace=True: re-rendering an unposted export (new hook, new cut) replaces it
                        res = pipeline.render_candidates(conn, cfg, src["id"], [c["id"]], platforms,
                                                         hook=hook or None, layout_kind=layout_kind, replace=True, log=st.caption)
                    done = [r for r in res if r["path"] and not r["blocked_reason"]]
                    st.success(f"{len(done)} file(s) rendered")
                    for r in res:
                        if r["blocked_reason"]:
                            st.warning(f"{r['platform']}: {r['blocked_reason']}")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        with right:
            if st.toggle("Preview", key=f"prev_{c['id']}"):
                with st.spinner("Cutting a preview ..."):
                    if 0 <= start < end <= src["duration"]:
                        proxy = pipeline.preview_proxy(cfg, src, dict(c, start_s=start, end_s=end), layout_kind)
                        st.video(str(proxy))
                    else:
                        st.warning("Check the preview bounds.")
            for e in db.list_exports(conn, candidate_id=c["id"]):
                note = f" - BLOCKED: {e['blocked_reason']}" if e["blocked_reason"] else ""
                post = db.post_for(conn, e["id"])
                note += " - posted" if post else ""
                if st.toggle(f"{e['platform']} export #{e['id']}{note}", key=f"exp_{e['id']}"):
                    st.video(e["path"])
                    st.code(e.get("caption") or "", language=None)
                    if post:
                        st.caption(f"Posted: {post['url']}")
                    else:
                        url = st.text_input("Post link", key=f"url_{e['id']}", placeholder="https://...")
                        if st.button("Mark as posted", key=f"posted_{e['id']}", disabled=not url.strip()):
                            db.add_post(conn, e["id"], url.strip())
                            st.rerun()
