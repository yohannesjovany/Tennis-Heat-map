"""
Unit tests for the ModelRegistry.
"""
import pytest

from tennis_heatmap.core.registry import ModelRegistry


class FakeBase:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_register_and_build():
    registry = ModelRegistry("TestRegistry")

    @registry.register("foo")
    class Foo(FakeBase):
        pass

    instance = registry.build("foo", x=1, y=2)
    assert isinstance(instance, Foo)
    assert instance.kwargs == {"x": 1, "y": 2}


def test_list_available():
    registry = ModelRegistry("TestRegistry")

    @registry.register("alpha")
    class A(FakeBase):
        pass

    @registry.register("beta")
    class B(FakeBase):
        pass

    assert registry.list_available() == ["alpha", "beta"]


def test_build_unknown_raises_key_error():
    registry = ModelRegistry("TestRegistry")
    with pytest.raises(KeyError, match="not_registered"):
        registry.build("not_registered")


def test_is_registered():
    registry = ModelRegistry("TestRegistry")

    @registry.register("bar")
    class Bar(FakeBase):
        pass

    assert registry.is_registered("bar") is True
    assert registry.is_registered("baz") is False


def test_overwrite_logs_warning(caplog):
    import logging
    registry = ModelRegistry("TestRegistry")

    @registry.register("dup")
    class First(FakeBase):
        pass

    with caplog.at_level(logging.WARNING):
        @registry.register("dup")
        class Second(FakeBase):
            pass

    assert "Overwriting" in caplog.text
    # The second registration should win
    assert registry.build("dup").__class__.__name__ == "Second"
