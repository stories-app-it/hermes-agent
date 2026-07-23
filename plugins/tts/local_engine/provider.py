"""Generic HTTP bridge to a locally-hosted TTS engine server.

One provider class serves any local engine (Sibilia-TTS today; Azzurra-
Voice, Kokoro-82M/Pocket-TTS, ChatTTS, F5-TTS are documented future
candidates) — the only per-engine config is ``endpoint`` in the
profile's ``config.yaml``. No engine-specific code lives here.
"""

from __future__ import annotations

from typing import Any, Optional

import requests

from agent.tts_provider import DEFAULT_OUTPUT_FORMAT, TTSProvider

_DEFAULT_TIMEOUT_SECONDS = 60


class LocalEngineTTSProvider(TTSProvider):
    """Routes synthesis to a local engine server over HTTP POST."""

    @property
    def name(self) -> str:
        return "local_engine"

    @property
    def display_name(self) -> str:
        return "Local Engine (self-hosted)"

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = DEFAULT_OUTPUT_FORMAT,
        **extra: Any,
    ) -> str:
        params = dict(extra)
        endpoint = params.pop("endpoint", None)
        if not endpoint:
            raise ValueError(
                "local_engine provider requires 'endpoint' "
                "(tts.local_engine.endpoint in config.yaml)"
            )
        timeout = params.pop("timeout", _DEFAULT_TIMEOUT_SECONDS)
        params.setdefault("model", model)
        params.setdefault("speed", speed)
        params.setdefault("format", format)

        try:
            response = requests.post(
                endpoint,
                json={"text": text, "voice": voice, "params": params},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"local_engine: request to {endpoint!r} failed: {exc}"
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"local_engine: {endpoint!r} returned HTTP "
                f"{response.status_code}: {response.text[:500]}"
            )

        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path
