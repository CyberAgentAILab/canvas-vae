import hashlib
import logging

logger = logging.getLogger(__name__)


def stable_hash(value):
    """Return a stable hash."""
    return int(hashlib.md5(str(value).encode('utf-8')).hexdigest(), 16)
