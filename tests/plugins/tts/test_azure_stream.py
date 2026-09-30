"""Sintesi Azure a pezzi (`synthesize_stream`) e SSML fissato.

Stories vuole mandare al browser i primi byte della prima frase mentre Azure
sta ancora producendo il resto. Qui si verifica che:

- l'SSML di `synthesize` non sia cambiato dopo l'estrazione di
  `_costruisci_ssml` (stringa attesa scritta a mano, non ricalcolata);
- `synthesize_stream` faccia la stessa POST, in streaming, sulla sessione
  condivisa, e consegni i byte nell'ordine in cui arrivano.
"""

from __future__ import annotations

import pytest

from plugins.tts.azure.provider import AzureSpeechTTSProvider

SSML_ATTESO = (
    '<speak version="1.0" xml:lang="it-IT" xmlns:mstts="https://www.w3.org/2001/mstts">'
    '<voice name="it-IT-Luca:MAI-Voice-2">'
    '<mstts:express-as style="cheerful" styledegree="1.50">'
    '<prosody rate="0.90" pitch="+2%" volume="80">'
    'Ciao.<break time="300ms"/>Come stai?<break time="600ms"/>Tutto bene.'
    "</prosody></mstts:express-as></voice></speak>"
)

PARAMETRI_SSML = dict(
    voice="it-IT-Luca:MAI-Voice-2",
    speed=0.9,
    pitch=2,
    volume=80,
    style="cheerful",
    style_degree=1.5,
    pause_sentence_ms=300,
    pause_paragraph_ms=600,
)
TESTO = "Ciao. Come stai?\n\nTutto bene."


class _RispostaFinta:
    def __init__(self, status_code=200, pezzi=(b"aaa", b"bbb", b"ccc")):
        self.status_code = status_code
        self._pezzi = list(pezzi)
        self.content = b"".join(self._pezzi)
        self.text = "errore finto"
        self.chiusa = False
        self.chunk_size_richiesto = None

    def iter_content(self, chunk_size=1):
        self.chunk_size_richiesto = chunk_size
        yield from self._pezzi

    def close(self):
        self.chiusa = True


class _CookieJarFinto:
    def set_policy(self, policy):  # noqa: ANN001
        pass


class _SessioneFinta:
    def __init__(self):
        self.richieste: list[dict] = []
        self.cookies = _CookieJarFinto()
        self.risposta = _RispostaFinta()

    def mount(self, prefisso, adapter):  # noqa: ANN001
        pass

    def post(self, url, **kwargs):  # noqa: ANN001
        self.richieste.append({"url": url, **kwargs})
        return self.risposta


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(
        "plugins.tts.azure.provider.requests.Session", _SessioneFinta, raising=True
    )
    monkeypatch.setattr(
        "plugins.tts.azure.provider.AzureSpeechTTSProvider._credentials",
        lambda self: ("chiave-finta", "northeurope"),
        raising=True,
    )
    return AzureSpeechTTSProvider()


def test_synthesize_produce_lo_stesso_ssml_di_prima(provider, tmp_path):
    provider.synthesize(TESTO, str(tmp_path / "o.mp3"), **PARAMETRI_SSML)
    inviato = provider._sessione().richieste[0]["data"].decode("utf-8")
    assert inviato == SSML_ATTESO


def test_ssml_estratto_e_lo_stesso(provider):
    assert provider._costruisci_ssml(TESTO, **PARAMETRI_SSML) == SSML_ATTESO


def test_stream_chiama_on_chunk_tre_volte_in_ordine(provider):
    ricevuti: list[bytes] = []
    provider.synthesize_stream(TESTO, on_chunk=ricevuti.append, **PARAMETRI_SSML)

    sessione = provider._sessione()
    assert ricevuti == [b"aaa", b"bbb", b"ccc"]
    assert len(sessione.richieste) == 1
    richiesta = sessione.richieste[0]
    assert richiesta["stream"] is True
    assert richiesta["data"].decode("utf-8") == SSML_ATTESO
    assert richiesta["url"] == (
        "https://northeurope.tts.speech.microsoft.com/cognitiveservices/v1"
    )
    assert richiesta["headers"]["Ocp-Apim-Subscription-Key"] == "chiave-finta"
    assert richiesta["headers"]["X-Microsoft-OutputFormat"] == "audio-24khz-160kbitrate-mono-mp3"
    assert b"".join(ricevuti) == sessione.risposta.content


def test_stream_chiude_la_risposta_e_riusa_la_sessione(provider):
    provider.synthesize_stream("Uno.", on_chunk=lambda c: None)
    provider.synthesize_stream("Due.", on_chunk=lambda c: None)
    sessione = provider._sessione()
    assert len(sessione.richieste) == 2
    assert sessione.risposta.chiusa is True


def test_stream_chiude_la_risposta_se_on_chunk_solleva(provider):
    def rompi(_):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        provider.synthesize_stream("Uno.", on_chunk=rompi)
    assert provider._sessione().risposta.chiusa is True


def test_stream_errore_http_solleva_e_non_consegna_nulla(provider):
    provider._sessione().risposta = _RispostaFinta(status_code=401)
    ricevuti: list[bytes] = []
    with pytest.raises(RuntimeError, match="401"):
        provider.synthesize_stream("Uno.", on_chunk=ricevuti.append)
    assert ricevuti == []
    assert provider._sessione().risposta.chiusa is True
