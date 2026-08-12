"""The checks a build runs before it emits anything.

Each check lives with the code that fails on the problem, and this module only
collects them. Running them up front means a build that cannot succeed fails
early, with a message about the input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from semanticlog.core.vocabulary import (
    SAFE_LOCAL_NAME as SAFE_LOCAL_NAME,
)
from semanticlog.core.vocabulary import (
    NameCollisionError as NameCollisionError,
)
from semanticlog.core.vocabulary import (
    UnsafeLocalNameError as UnsafeLocalNameError,
)
from semanticlog.data.integrity import require_integrity

if TYPE_CHECKING:
    from collections.abc import Callable

    from semanticlog.data.oced import OCEDStore
    from semanticlog.onto.settings import OntoBuildSettings


def input_checks(settings: OntoBuildSettings) -> list[Callable[[OCEDStore], None]]:
    """Return the checks that apply to a build with these settings.

    Args:
        settings: Settings of the build, which decide whether the checks that
            only apply to some representations do anything.

    Returns:
        Callables that raise if the store is unfit for this build.
    """
    from semanticlog.onto.emitters.attributes import AttributeEmitter
    from semanticlog.onto.emitters.order import OrderEmitter

    return [
        require_integrity,
        lambda data: OrderEmitter.check_declare_input(data, settings),
        AttributeEmitter.check_converging_values,
    ]


def audit_build(data: OCEDStore, settings: OntoBuildSettings) -> None:
    """Run every applicable check, stopping at the first failure.

    Raises:
        ValueError: If a check rejects the store.
    """
    for check in input_checks(settings):
        check(data)
