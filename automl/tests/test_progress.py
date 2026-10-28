from io import StringIO

from automl.progress import ProgressEvent, StderrProgress, format_progress


class FlushStream(StringIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1
        super().flush()


def test_stderr_progress_flushes_each_event():
    stream = FlushStream()
    reporter = StderrProgress(stream)
    reporter(ProgressEvent("tuning", "trial.complete", 1.25, {"trial": 2, "score": 0.8}))

    assert stream.flush_count == 1
    assert "[tuning] completed trial trial=2 score=0.8 elapsed=1.2s" in stream.getvalue()


def test_progress_format_includes_best_score_and_direction():
    line = format_progress(
        ProgressEvent(
            "tuning",
            "trial.complete",
            3.0,
            {"trial": 4, "score": 0.7, "best_score": 0.7, "direction": "maximize"},
        )
    )

    assert "score=0.7" in line
    assert "best_score=0.7" in line
    assert "direction=maximize" in line
    assert "elapsed=3.0s" in line
