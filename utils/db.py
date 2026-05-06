import logging
import os
from pathlib import Path

import psycopg2
import psycopg2.pool
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

log = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "skincare_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool | None:
    global _pool
    if _pool is not None and not _pool.closed:
        return _pool
    try:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            cursor_factory=RealDictCursor,
            keepalives=1,
            keepalives_idle=60,
            keepalives_interval=10,
            keepalives_count=5,
            **DB_CONFIG,
        )
        return _pool
    except psycopg2.OperationalError as e:
        log.error("Could not connect to database: %s", e)
        return None


def run_query(query: str, params: tuple = None) -> list[dict]:
    pool = _get_pool()
    if pool is None:
        return []
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        log.error("Query error: %s", e)
        try:
            if conn:
                conn.rollback()
        except Exception:
            pass
        return []
    finally:
        if conn:
            pool.putconn(conn)


def run_mutation(query: str, params: tuple = None) -> bool:
    pool = _get_pool()
    if pool is None:
        return False
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        log.error("Mutation error: %s", e)
        try:
            if conn:
                conn.rollback()
        except Exception:
            pass
        return False
    finally:
        if conn:
            pool.putconn(conn)


def check_connection() -> bool:
    pool = _get_pool()
    if pool is None:
        return False
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False
    finally:
        if conn:
            pool.putconn(conn)
