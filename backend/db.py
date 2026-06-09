from functools import lru_cache

from supabase import Client, create_client

from config import SUPABASE_KEY, SUPABASE_URL


def is_supabase_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


@lru_cache
def get_supabase() -> Client:
    if not is_supabase_configured():
        raise RuntimeError(
            "Database not configured. Set SUPABASE_URL and SUPABASE_KEY in .env "
            "(copy from .env.example). Run supabase/migrations SQL files in Supabase."
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)
