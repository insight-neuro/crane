from crane.data.selectors import (
    _parse_stem,
    _get_token,
    SelectAll,
    SelectNone,
    Subjects,
    SubjectSessions,
    Suffix,
    Entity,
)


STEM = "sub-001_ses-01_task-rest_run-02_bold"


# -------------------------
# parsing helpers
# -------------------------

def test_parse_stem():
    parts = _parse_stem(STEM)
    assert parts == [
        "sub-001",
        "ses-01",
        "task-rest",
        "run-02",
        "bold",
    ]


def test_get_token():
    parts = _parse_stem(STEM)
    assert _get_token(parts, "sub-") == "001"
    assert _get_token(parts, "ses-") == "01"
    assert _get_token(parts, "run-") == "02"
    assert _get_token(parts, "foo-") is None


# -------------------------
# trivial selectors
# -------------------------

def test_select_all_none():
    assert SelectAll().match(STEM)
    assert not SelectNone().match(STEM)


# -------------------------
# subjects
# -------------------------

def test_subjects_match():
    sel = Subjects(1)
    assert sel.match(STEM)

    sel = Subjects("sub-001")
    assert sel.match(STEM)

    sel = Subjects(2)
    assert not sel.match(STEM)


def test_subject_sessions_match():
    sel = SubjectSessions((1, 1))
    assert sel.match(STEM)

    sel = SubjectSessions(("001", "01"))
    assert sel.match(STEM)

    sel = SubjectSessions((1, 2))
    assert not sel.match(STEM)


# -------------------------
# suffix
# -------------------------

def test_suffix():
    assert Suffix("task-rest").match(STEM)
    assert not Suffix("task-motor").match(STEM)


# -------------------------
# entity
# -------------------------

def test_entity_key_only():
    assert Entity("task").match(STEM)
    assert Entity("run").match(STEM)
    assert not Entity("acq").match(STEM)


def test_entity_key_value():
    assert Entity("task", "rest").match(STEM)
    assert Entity("run", "02").match(STEM)
    assert not Entity("run", "01").match(STEM)


# -------------------------
# logical operators
# -------------------------

def test_and_or_not():
    sel = Subjects(1) & Entity("task", "rest")
    assert sel.match(STEM)

    sel = Subjects(1) & Entity("task", "motor")
    assert not sel.match(STEM)

    sel = Subjects(1) | Subjects(2)
    assert sel.match(STEM)

    sel = ~Subjects(2)
    assert sel.match(STEM)
    assert not (~Subjects(1)).match(STEM)


def test_xor():
    sel = Subjects(1) ^ Entity("run", "01")
    # Subjects(1)=True, run-01=False → True XOR False → True
    assert sel.match(STEM)


def test_sub_operator():
    sel = Subjects(1) - Entity("run", "02")
    # True & ~True → False
    assert not sel.match(STEM)
