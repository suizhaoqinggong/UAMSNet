from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeAlias, TypeVar

T = TypeVar("T")
Factory: TypeAlias = Callable[..., T]


class RegistryError(ValueError):
    pass


class DuplicateRegistrationError(RegistryError):
    pass


class UnknownRegistrationError(RegistryError):
    pass


def normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Component name cannot be empty.")
    return normalized


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        if not kind.strip():
            raise ValueError("Registry kind cannot be empty.")
        self._kind = kind
        self._factories: dict[str, Factory[T]] = {}

    @property
    def kind(self) -> str:
        return self._kind

    def register(self, name: str, factory: Factory[T]) -> None:
        normalized = normalize_name(name)
        if normalized in self._factories:
            message = f"{self._kind} '{normalized}' is already registered."
            raise DuplicateRegistrationError(message)
        self._factories[normalized] = factory

    def decorator(self, name: str) -> Callable[[Factory[T]], Factory[T]]:
        def _register(factory: Factory[T]) -> Factory[T]:
            self.register(name=name, factory=factory)
            return factory

        return _register

    def get(self, name: str) -> Factory[T]:
        normalized = normalize_name(name)
        try:
            return self._factories[normalized]
        except KeyError as exc:
            available = ", ".join(self.names()) if self._factories else "<none>"
            message = f"{self._kind} '{normalized}' is not registered. Available: {available}"
            raise UnknownRegistrationError(message) from exc

    def create(self, name: str, **kwargs: object) -> T:
        factory = self.get(name)
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return normalize_name(name) in self._factories
