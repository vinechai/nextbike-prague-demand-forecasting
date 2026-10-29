"""database connection pool for pipeline scripts."""

import os
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

_POOL: ThreadedConnectionPool | None = None


def _pool() -> ThreadedConnectionPool:
    global _POOL
    if _POOL is None:
        _POOL = ThreadedConnectionPool(
            1, 5,
            host=os.environ.get("DB_HOST", "localhost"),
            port=int(os.environ.get("DB_PORT", "5432")),
            user=os.environ.get("DB_USER", "nextbike"),
            password=os.environ.get("DB_PASS", "nextbike"),
            dbname=os.environ.get("DB_NAME", "nextbike"),
        )
    return _POOL


@contextmanager
def get_conn():
    conn = _pool().getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool().putconn(conn)
