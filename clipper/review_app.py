"""Review app (Streamlit): preview candidates, label them good or bad, render the good ones.

Start it with `clipper review`. Labels are what `clipper eval` and `clipper tune` learn from.
"""

from __future__ import annotations

import streamlit as st

from clipper import config, db, evaluate, media, pipeline

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
cands = db.list_candidates(conn, src["id"])
labels = db.list_labels(conn, src["id"])
show = st.sidebar.radio("Show", ["all", "not reviewed", "good", "bad"])
platforms = st.sidebar.multiselect("Platforms to render", list(cfg["platforms"]), default=list(cfg["platforms"]))
st.sidebar.caption(f"Rights: {src['rights']}")

if not cands:
    st.warning(f"No candidates for this video yet: run `clipper analyze {src['id']}`.")
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
            st.markdown(f"**#{c['id']} - rank {c['rank']}** - {media.fmt_ts(c['start_s'])} -> "
                        f"{media.fmt_ts(c['end_s'])} - {c['end_s'] - c['start_s']:.0f}s - score {c['score']:.2f} - "
                        f"*{LABEL_NAMES[lab]}*")
            sig = c.get("signals") or {}
            st.caption(" - ".join(f"{k} {v:.1f}" for k, v in sorted(sig.items(), key=lambda kv: -kv[1]) if v > 0)
                       or "no signal detail")
            if (c.get("flags") or {}).get("spoiler_review"):
                st.warning("Looks like a cutscene: check for story spoilers before posting.")
            if c.get("text"):
                st.write(f"\"{c['text'][:300]}\"")
            hook = st.text_input("Hook", value=c.get("hook") or "", key=f"hook_{c['id']}")
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
                if hook != (c.get("hook") or ""):
                    db.set_hook(conn, c["id"], hook)
                try:
                    with st.spinner("Rendering ..."):
                        # replace=True: re-rendering an unposted export (new hook, new cut) replaces it
                        res = pipeline.render_candidates(conn, cfg, src["id"], [c["id"]], platforms,
                                                         hook=hook or None, replace=True, log=st.caption)
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
                    proxy = pipeline.preview_proxy(cfg, src, c)
                st.video(str(proxy))
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
