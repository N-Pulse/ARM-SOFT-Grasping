"""Dataset capture contracts shared by every vision workflow."""

from .session import CaptureSessionWriter, SESSION_SCHEMA_VERSION

__all__ = ["CaptureSessionWriter", "SESSION_SCHEMA_VERSION"]
