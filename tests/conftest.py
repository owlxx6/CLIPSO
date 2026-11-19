import pytest


@pytest.fixture(scope="session")
def fx(tmp_path_factory):
    """A 3-minute synthetic stream (small resolution to keep tests fast) with known highlights."""
    from clipper import fixture

    return fixture.make_fixture(tmp_path_factory.mktemp("fixture"), duration=180.0, size=(640, 360))


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    """Defaults only, with data/ and config/ isolated in a temp folder."""
    monkeypatch.setenv("CLIPPER_HOME", str(tmp_path))
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("CLIPPER_CONFIG", raising=False)
    from clipper import config

    return config.load_config()


@pytest.fixture()
def source(cfg, fx):
    """The fixture stream ingested into a fresh database, with the facecam box set by hand."""
    from clipper import db, pipeline

    conn = db.connect(cfg)
    sid = pipeline.ingest(conn, cfg, fx["video"], rights="test:synthetic", title="fixture",
                          chat_file=fx["chat"], clips_file=fx["clips"], transcript_file=fx["transcript"])
    pipeline.set_layout(cfg, db.get_source(conn, sid), "stacked", fx["facecam"])
    return conn, sid
