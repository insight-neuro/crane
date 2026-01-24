from .base import NeuralData, NeuralLabeledData


class SubjectSession(NeuralData):
    """Neural data from a single subject session."""

    subject_id: str
    """Identifier for the subject."""
    session_id: str
    """Identifier for the session."""


class SubjectSessionLabeled(NeuralLabeledData, SubjectSession):
    """Neural data with labels from a single subject session."""
