"""
FastAPI dependency for extracting and validating the current user
from a Supabase JWT in the Authorization header.
"""
from fastapi import Header, HTTPException

from auth.supabase_client import get_supabase


class CurrentUser:
    def __init__(self, id: str, email: str, role: str, display_name: str):
        self.id = id
        self.email = email
        self.role = role
        self.display_name = display_name


async def get_current_user(authorization: str = Header(None)) -> CurrentUser:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.removeprefix("Bearer ")
    supabase = get_supabase()

    try:
        user_response = supabase.auth.get_user(token)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        profile = (
            supabase.table("users")
            .select("*")
            .eq("id", user.id)
            .single()
            .execute()
        )

        if not profile.data:
            raise HTTPException(status_code=404, detail="User profile not found")

        return CurrentUser(
            id=user.id,
            email=user.email,
            role=profile.data.get("role", "student"),
            display_name=profile.data.get("display_name", ""),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth error: {str(e)}")
