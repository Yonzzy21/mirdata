"""Tests for the Vienna 4x22 loader."""

from mirdata.annotations import NoteData
import logging
import os

try:
    import partitura
except ImportError:
    logging.error(
        "In order to test vienna4x22 you must have partitura installed. "
        "Please reinstall mirdata using `pip install 'mirdata[vienna4x22]'` "
        "and re-run the tests."
    )
    raise ImportError

import numpy as np

from mirdata.datasets import vienna4x22
from tests.test_utils import run_track_tests

DATA_HOME = os.path.normpath("tests/resources/mir_datasets/vienna4x22")
TRACK_ID = "Chopin_op10_no3_p01"


def test_track():
    dataset = vienna4x22.Dataset(DATA_HOME, version="test")
    track = dataset.track(TRACK_ID)

    expected_attributes = {
        "track_id": TRACK_ID,
        "piece": "Chopin_op10_no3",
        "pianist_id": "01",
        "alignment_quality": "manual",
        "audio_path": os.path.join(
            DATA_HOME, "audio/Chopin_Etude/Chopin_op10_no3_p01.wav"
        ),
        "score_path": os.path.join(DATA_HOME, "musicxml/Chopin_op10_no3.musicxml"),
        "performance_path": os.path.join(DATA_HOME, "midi/Chopin_op10_no3_p01.mid"),
        "match_path": os.path.join(DATA_HOME, "match/Chopin_op10_no3_p01.match"),
    }

    expected_property_types = {
        "audio": tuple,
        "score": partitura.score.Score,
        "performance": partitura.performance.Performance,
        "match": tuple,
        "score_note_array": np.ndarray,  # Changing for the test and then fix loader to fit score_note_array
        "performance_note_array": NoteData,  ###change from ndarray to NoteData,add the content we're expecting [(-0.5 , 0.5 , -0.5 , 0.5 ,   0,  8, 59, 1, 'n1', 16)
    }

    run_track_tests(track, expected_attributes, expected_property_types)


def test_load_audio():
    path = os.path.join(DATA_HOME, "audio/Chopin_Etude/Chopin_op10_no3_p01.wav")
    audio, sr = vienna4x22.load_audio(path)
    assert audio.shape[0] == 2  # stereo
    assert sr == 44100
    assert vienna4x22.load_audio(None) is None


def test_load_score():
    path = os.path.join(DATA_HOME, "musicxml/Chopin_op10_no3.musicxml")
    score = vienna4x22.load_score(path)
    assert isinstance(score, partitura.score.Score)
    assert vienna4x22.load_score(None) is None
    na = score.note_array()
    assert na.shape[0] > 0  # check not empty
    # checking the first row fits
    # Exact checks for discrete fields (strings, ints):
    assert na[0]["id"] == "n1"
    assert na[0]["pitch"] == 59
    assert na[0]["voice"] == 1
    # Float timing fields
    assert np.isclose(na[0]["onset_beat"], -0.5)
    assert np.isclose(na[0]["duration_beat"], 0.5)


###test for the partitura performance format
def test_load_performance():
    path = os.path.join(DATA_HOME, "midi/Chopin_op10_no3_p01.mid")
    perf = vienna4x22.load_performance(path)
    assert isinstance(perf, partitura.performance.Performance)
    na = perf.note_array()
    assert na.shape[0] > 0
    # checking the first row fits
    # Exact checks for discrete fields (strings, ints):
    assert na[0]["id"] == "n0"
    assert na[0]["pitch"] == 59
    assert na[0]["velocity"] == 44
    assert np.isclose(na[0]["onset_sec"], 0.0)
    assert np.isclose(na[0]["duration_sec"], 0.87395835)
    assert vienna4x22.load_performance(None) is None


def test_load_performance_note_array():
    path = os.path.join(DATA_HOME, "midi/Chopin_op10_no3_p01.mid")
    note_data = vienna4x22.load_performance_note_array(path)

    assert isinstance(note_data, NoteData)
    assert note_data.interval_unit == "s"
    assert note_data.pitch_unit == "midi"
    assert note_data.intervals.shape == (451, 2)
    assert len(note_data.pitches) == 451
    # Check first note onset, offset, and pitch
    assert np.isclose(note_data.intervals[0, 0], 0.0)
    assert np.isclose(note_data.intervals[0, 1], 0.87395835)
    assert note_data.pitches[0] == 59
    # Test None handling
    assert vienna4x22.load_performance_note_array(None) is None


def test_load_match():
    path = os.path.join(DATA_HOME, "match/Chopin_op10_no3_p01.match")
    result = vienna4x22.load_match(path)
    assert isinstance(result, tuple)
    assert len(result) == 3
    performance, alignment, score = result
    assert isinstance(performance, partitura.performance.Performance)
    assert isinstance(alignment, list) and len(alignment) > 0
    # Check first alignment entry
    assert alignment[0]["label"] == "match"
    assert alignment[0]["score_id"] == "n1"
    assert alignment[0]["performance_id"] == "n0"
    assert isinstance(score, partitura.score.Score)
    assert vienna4x22.load_match(None) is None
