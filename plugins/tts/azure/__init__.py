"""Microsoft Speech (Azure AI Speech) TTS plugin — bundled, auto-loaded.

Mirrors the ``plugins/tts/local_engine/`` layout: ``provider.py`` holds
the provider class; ``__init__.py::register`` instantiates and
registers it.
"""

from __future__ import annotations

from plugins.tts.azure.provider import AzureSpeechTTSProvider


def register(ctx) -> None:
    """Register the Azure Speech TTS provider with the plugin context."""
    ctx.register_tts_provider(AzureSpeechTTSProvider())
