"""Generic local-engine TTS plugin — bundled, auto-loaded.

Mirrors the ``plugins/browser/<vendor>/`` layout: ``provider.py`` holds
the provider class; ``__init__.py::register`` instantiates and
registers it.
"""

from __future__ import annotations

from plugins.tts.local_engine.provider import LocalEngineTTSProvider


def register(ctx) -> None:
    """Register the local-engine TTS provider with the plugin context."""
    ctx.register_tts_provider(LocalEngineTTSProvider())
