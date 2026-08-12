"""Fixtures shared by the ontology tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from semanticlog.onto.settings import DEFAULT_IRI, OntoBuildSettings
from tests.onto.support.recording import RecordingSink


@pytest.fixture
def recording_sink() -> Callable[..., RecordingSink]:

    def make(settings: OntoBuildSettings, iri: str = DEFAULT_IRI) -> RecordingSink:
        return RecordingSink(settings, iri)

    return make
