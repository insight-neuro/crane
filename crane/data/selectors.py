from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


def _parse_stem(stem: str) -> list[str]:
    # stem is filename without extension, e.g. "sub-001_ses-01_task-rest_run-01_bold"
    return stem.split("_")


def _get_token(parts: list[str], prefix: str) -> str | None:
    # prefix like "sub-" or "ses-"
    for p in parts:
        if p.startswith(prefix):
            return p[len(prefix) :]
    return None


class Selector(ABC):
    @abstractmethod
    def match(self, stem: str) -> bool: ...

    def __and__(self, other: Selector) -> Selector:
        return _AndSelector(self, other)

    def __or__(self, other: Selector) -> Selector:
        return _OrSelector(self, other)

    def __invert__(self) -> Selector:
        return _NotSelector(self)

    def __xor__(self, other: Selector) -> Selector:
        return (self & ~other) | (~self & other)

    def __sub__(self, other: Selector) -> Selector:
        return self & ~other


class SelectAll(Selector):
    def match(self, stem: str) -> bool:
        return True


class SelectNone(Selector):
    def match(self, stem: str) -> bool:
        return False


@dataclass(frozen=True)
class _AndSelector(Selector):
    left: Selector
    right: Selector

    def match(self, stem: str) -> bool:
        return self.left.match(stem) and self.right.match(stem)


@dataclass(frozen=True)
class _OrSelector(Selector):
    left: Selector
    right: Selector

    def match(self, stem: str) -> bool:
        return self.left.match(stem) or self.right.match(stem)


@dataclass(frozen=True)
class _NotSelector(Selector):
    selector: Selector

    def match(self, stem: str) -> bool:
        return not self.selector.match(stem)


class Subjects(Selector):
    def __init__(self, *subject_ids: int | str) -> None:
        self.subject_ids = {str(s).replace("sub-", "").zfill(3) for s in subject_ids}

    def match(self, stem: str) -> bool:
        parts = _parse_stem(stem)
        sub = _get_token(parts, "sub-")
        return (sub is not None) and (sub in self.subject_ids)


class SubjectSessions(Selector):
    def __init__(self, *pairs: tuple[int | str, int | str]) -> None:
        self.pairs = {
            (str(sub).replace("sub-", "").zfill(3), str(ses).replace("ses-", "").zfill(2)) for sub, ses in pairs
        }

    def match(self, stem: str) -> bool:
        parts = _parse_stem(stem)
        sub = _get_token(parts, "sub-")
        ses = _get_token(parts, "ses-")
        return (sub is not None) and (ses is not None) and ((sub, ses) in self.pairs)


class Suffix(Selector):
    """Requires an exact suffix to be present, e.g. 'task-rest'."""

    def __init__(self, token: str) -> None:
        self.token = token

    def match(self, stem: str) -> bool:
        parts = _parse_stem(stem)
        return self.token in parts


class Entity(Selector):
    """Match BIDS-style entity key-value tokens like task-rest, run-01."""

    def __init__(self, key: str, value: str | None = None) -> None:
        self.key = key
        self.value = value

    def match(self, stem: str) -> bool:
        parts = _parse_stem(stem)
        prefix = f"{self.key}-"
        for p in parts:
            if p.startswith(prefix):
                if self.value is None:
                    return True
                _, v = p.split("-", 1)
                return v == self.value
        return False
