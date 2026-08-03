"""Microsoft Speech (Azure AI Speech) TTS provider.

REST-only integration (no ``azure-cognitiveservices-speech`` SDK dependency):
posts SSML to the regional Cognitive Services TTS endpoint and writes the
returned audio bytes to ``output_path``. Mirrors the curl-based verification
flow documented in ``analisi_tts/Stories_tts_setup_provider.md``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

import requests

from agent.tts_provider import DEFAULT_OUTPUT_FORMAT, TTSProvider
from hermes_cli.config import get_env_value

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 30
_DEFAULT_VOICE = "it-IT-ElsaNeural"
_DEFAULT_LOCALE = "it-IT"
_MSTTS_NAMESPACE = "https://www.w3.org/2001/mstts"

# Uniche voci con StyleList reale (confermato dal vivo, vedi
# backend/tts_catalog.py) — <mstts:express-as> viene applicato solo su
# queste, altrimenti Microsoft droppa in silenzio l'intero elemento e la
# richiesta torna alla voce neutra senza errore (analisi_tts/
# Stories_tts_ab_test.md §3.4): meglio ignorare noi lo style qui che farlo
# fallire in silenzio lato Microsoft.
_STYLE_CAPABLE_VOICES = {
    "it-IT-Luca:MAI-Voice-2",
    "it-IT-Luca:MAI-Voice-2-Flash",
    "it-IT-Rosa:MAI-Voice-2",
    "it-IT-Rosa:MAI-Voice-2-Flash",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Formato di output richiesto -> header X-Microsoft-OutputFormat Azure.
# Azure non ha un formato flac nativo: per quel caso si ricade su wav (il
# chiamante ri-transcodifica se serve, come per gli altri provider REST).
_OUTPUT_FORMAT_MAP = {
    "mp3": "audio-24khz-160kbitrate-mono-mp3",
    "wav": "riff-24khz-16bit-mono-pcm",
    "ogg": "ogg-24khz-16bit-mono-opus",
    "opus": "ogg-24khz-16bit-mono-opus",
    "flac": "riff-24khz-16bit-mono-pcm",
}


class AzureSpeechTTSProvider(TTSProvider):
    """Routes synthesis to Azure AI Speech's REST TTS endpoint."""

    @property
    def name(self) -> str:
        return "azure"

    @property
    def display_name(self) -> str:
        return "Microsoft Speech"

    def is_available(self) -> bool:
        return bool(get_env_value("AZURE_SPEECH_KEY")) and bool(
            get_env_value("AZURE_SPEECH_REGION")
        )

    def _credentials(self) -> tuple[str, str]:
        key = get_env_value("AZURE_SPEECH_KEY")
        region = get_env_value("AZURE_SPEECH_REGION")
        if not key or not region:
            raise ValueError(
                "AZURE_SPEECH_KEY/AZURE_SPEECH_REGION non impostate. "
                "Vedi analisi_tts/Stories_tts_setup_provider.md."
            )
        return key, region

    def list_voices(self) -> List[Dict[str, Any]]:
        """Interroga live l'endpoint voices/list — nessun elenco hardcoded.

        La disponibilità/stili delle voci cambia nel tempo (vedi doc di
        setup, sezione 1.4): meglio una chiamata reale che una lista
        potenzialmente stale.
        """
        try:
            key, region = self._credentials()
        except ValueError:
            return []
        try:
            response = requests.get(
                f"https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list",
                headers={"Ocp-Apim-Subscription-Key": key},
                timeout=_DEFAULT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            voices = response.json()
        except (requests.RequestException, ValueError):
            return []

        return [
            {
                "id": v.get("ShortName"),
                "display": f"{v.get('ShortName')} ({v.get('Gender')})",
                "language": v.get("Locale"),
                "gender": (v.get("Gender") or "").lower(),
            }
            for v in voices
            if v.get("Locale", "").startswith("it-")
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "cloud",
            "tag": "Azure AI Speech — voci neurali Microsoft",
            "env_vars": [
                {
                    "key": "AZURE_SPEECH_KEY",
                    "prompt": "Azure Speech subscription key",
                    "url": "https://portal.azure.com",
                },
                {
                    "key": "AZURE_SPEECH_REGION",
                    "prompt": "Azure Speech region (es. westeurope)",
                    "url": "https://portal.azure.com",
                },
            ],
        }

    def _body_with_pauses(
        self,
        text: str,
        pause_sentence_ms: Optional[float],
        pause_paragraph_ms: Optional[float],
    ) -> str:
        """Weave <break> tags between paragraphs/sentences (each escaped on its own).

        Le pause del documento AB-test (§3.1, pause_sentence_ms/
        pause_paragraph_ms) sono pensate per un testo con struttura a
        frasi/paragrafi, ma il chiamante passa una singola stringa: qui la
        struttura viene ricavata da doppio a-capo (paragrafi) e punteggiatura
        di fine frase (frasi), senza dipendenze NLP aggiuntive.
        """
        paragraphs = [p for p in text.split("\n\n") if p.strip()] or [text]
        paragraph_break = (
            f'<break time="{int(pause_paragraph_ms)}ms"/>' if pause_paragraph_ms else ""
        )
        sentence_break = (
            f'<break time="{int(pause_sentence_ms)}ms"/>' if pause_sentence_ms else ""
        )

        paragraph_bodies = []
        for paragraph in paragraphs:
            sentences = [s for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
            paragraph_bodies.append(
                sentence_break.join(escape(s) for s in sentences) if sentence_break else escape(paragraph)
            )
        return paragraph_break.join(paragraph_bodies) if paragraph_break else "".join(paragraph_bodies)

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = DEFAULT_OUTPUT_FORMAT,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
        style: Optional[str] = None,
        style_degree: Optional[float] = None,
        pause_sentence_ms: Optional[float] = None,
        pause_paragraph_ms: Optional[float] = None,
        **extra: Any,
    ) -> str:
        key, region = self._credentials()
        voice_name = voice or _DEFAULT_VOICE
        output_format = _OUTPUT_FORMAT_MAP.get(format, _OUTPUT_FORMAT_MAP["mp3"])

        body = self._body_with_pauses(text, pause_sentence_ms, pause_paragraph_ms)

        prosody_attrs = []
        if speed is not None:
            prosody_attrs.append(f'rate="{speed:.2f}"')
        if pitch is not None:
            prosody_attrs.append(f'pitch="{pitch:+.0f}%"')
        if volume is not None:
            prosody_attrs.append(f'volume="{volume:.0f}"')
        if prosody_attrs:
            body = f'<prosody {" ".join(prosody_attrs)}>{body}</prosody>'

        use_style = bool(style) and voice_name in _STYLE_CAPABLE_VOICES
        mstts_namespace = ""
        if use_style:
            mstts_namespace = f' xmlns:mstts="{_MSTTS_NAMESPACE}"'
            degree_attr = f' styledegree="{style_degree:.2f}"' if style_degree is not None else ""
            body = f'<mstts:express-as style="{escape(style)}"{degree_attr}>{body}</mstts:express-as>'

        ssml = (
            f'<speak version="1.0" xml:lang="{_DEFAULT_LOCALE}"{mstts_namespace}>'
            f'<voice name="{escape(voice_name)}">{body}</voice>'
            f"</speak>"
        )

        logger.info(
            "azure tts: voice=%r speed=%r pitch=%r volume=%r style=%r "
            "style_degree=%r pause_sentence_ms=%r pause_paragraph_ms=%r ssml=%s",
            voice_name, speed, pitch, volume, style,
            style_degree, pause_sentence_ms, pause_paragraph_ms, ssml,
        )

        try:
            response = requests.post(
                f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
                data=ssml.encode("utf-8"),
                headers={
                    "Ocp-Apim-Subscription-Key": key,
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": output_format,
                    "User-Agent": "stories-mvp",
                },
                timeout=_DEFAULT_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"azure: request failed: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"azure: {region!r} returned HTTP "
                f"{response.status_code}: {response.text[:500]}"
            )

        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path
