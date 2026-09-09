"""
Supabase client singleton + auth helpers.
"""
import os
from supabase import create_client, Client

_client: Client = None


def get_supabase() -> Client:
    global _client
    if _client is not None:
        return _client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"
        )

    _client = create_client(url, key)
    return _client
