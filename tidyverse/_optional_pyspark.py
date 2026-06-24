"""Optional pyspark support.

pyspark is a heavy, Java-dependent dependency that the library uses only for type
annotations and a handful of ``isinstance`` checks. Importing it lazily here lets the
pandas code paths run (and be tested) without pyspark installed: when pyspark is
present the real ``pyspark.sql`` module and ``concat_ws`` are re-exported, otherwise
lightweight stand-in types are provided so annotations and ``isinstance`` checks keep
working. Any spark-only operation that actually needs pyspark raises a clear error.
"""

from types import SimpleNamespace

try:
    import pyspark.sql as ps
    from pyspark.sql.functions import concat_ws

    HAS_PYSPARK = True
except ModuleNotFoundError:
    HAS_PYSPARK = False

    # Distinct stand-in classes so isinstance(<pandas object>, ps.DataFrame) is False
    # and annotations referencing ps.DataFrame / ps.Column resolve at definition time.
    _DataFrame = type("DataFrame", (), {})
    _Column = type("Column", (), {})

    ps = SimpleNamespace(
        DataFrame=_DataFrame,
        Column=_Column,
        column=SimpleNamespace(Column=_Column),
    )

    def concat_ws(*args, **kwargs):
        """Raise a clear error: pyspark is required for this spark-only operation."""
        raise ModuleNotFoundError(
            "pyspark is required for this operation but is not installed. "
            "Install it with `pip install pyspark` to use the spark code paths."
        )


__all__ = ["ps", "concat_ws", "HAS_PYSPARK"]
