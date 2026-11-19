import importlib.util
from pathlib import Path

import pytest

from clipper import cli, db, pipeline


def test_doctor_runs(cfg, capsys):
    code = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert "ffmpeg" in out and "opencv" in out
    assert code in (0, 1)


def test_ingest_requires_rights(cfg, fx):
    with pytest.raises(SystemExit):
        cli.main(["ingest", str(fx["video"])])


def test_cli_flow(cfg, fx, capsys):
    assert cli.main(["ingest", str(fx["video"]), "--rights", "test:cli", "--chat", str(fx["chat"]),
                     "--clips", str(fx["clips"]), "--transcript", str(fx["transcript"])]) == 0
    assert cli.main(["analyze", "1", "--no-transcribe", "--top", "3"]) == 0
    assert cli.main(["label", "1", "--truth", str(fx["truth"])]) == 0
    assert cli.main(["eval", "1", "--k", "3"]) == 0
    out = capsys.readouterr().out
    assert "precision@3 = 1.0" in out


@pytest.mark.skipif(importlib.util.find_spec("streamlit") is None, reason="streamlit not installed")
def test_review_app_starts(cfg, source):
    from streamlit.testing.v1 import AppTest

    conn, sid = source
    pipeline.analyze(conn, cfg, sid, transcribe=False)
    at = AppTest.from_file(str(Path(cli.__file__).with_name("review_app.py")), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("rank 1" in m.value for m in at.markdown)
    at.button(key=f"good_{db.list_candidates(conn, sid)[0]['id']}").click().run()
    assert not at.exception
    assert any(row["label"] == 1 for row in db.list_labels(conn, sid))
