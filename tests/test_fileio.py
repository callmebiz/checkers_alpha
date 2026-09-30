import threading
import time

from training import fileio


def test_write_text_waits_for_a_reader_to_let_go(tmp_path):
    """On Windows a file can't be replaced while another process has it
    open; the write must retry rather than fail."""
    path = tmp_path / 'status.json'
    path.write_text('old', encoding='utf-8')
    reader = path.open('r', encoding='utf-8')
    threading.Timer(0.3, reader.close).start()
    assert fileio.write_text(path, 'new')
    assert path.read_text(encoding='utf-8') == 'new'


def test_optional_write_gives_up_quietly(tmp_path, monkeypatch):
    """An optional write gives up quickly and leaves no temp file."""
    monkeypatch.setattr(fileio, 'RETRY_SECONDS', 0.2)
    path = tmp_path / 'status.json'
    path.write_text('old', encoding='utf-8')
    with path.open('r', encoding='utf-8'):
        start = time.monotonic()
        ok = fileio.write_text(path, 'new', required=False)
        assert time.monotonic() - start < 2
    # on Windows the reader blocks the replace; elsewhere it succeeds
    assert ok == (path.read_text(encoding='utf-8') == 'new')
    assert not fileio.tmp_path(path).exists()


def test_read_jsonl_skips_a_partial_last_line(tmp_path):
    """A half-written last line is ignored rather than raising."""
    path = tmp_path / 'm.jsonl'
    fileio.append_jsonl(path, {'iteration': 0})
    with path.open('a', encoding='utf-8') as f:
        f.write('{"iteration": 1, "los')  # still being written
    assert fileio.read_jsonl(path) == [{'iteration': 0}]
