"""La connessione verso Azure si riusa fra una sintesi e l'altra.

Perche' questo test esiste
--------------------------
Il provider apriva una connessione nuova a ogni `synthesize()` (`requests.post`
diretto). Stories sintetizza **frase per frase**, non a risposta intera, quindi
l'apertura si pagava una volta per frase e non una per turno.

Misurato il 15-09-2026 verso `northeurope.tts.speech.microsoft.com`: aprire una
connessione costa ~117 ms di mediana (TCP ~60, TLS 1.3 ~58). Con 3-4 frasi per
risposta sono 350-470 ms per turno, circa un sesto del budget di sintesi.

La regressione che questi test sorvegliano e' silenziosa: tornando a
`requests.post` la sintesi **continua a funzionare**, nessun log lo dice, e si
perde solo tempo. E' la stessa famiglia di trappole descritta in
`stories_mvp/CLAUDE.md` per il commit del provider TTS.
"""

from __future__ import annotations

import threading

import pytest

from plugins.tts.azure.provider import AzureSpeechTTSProvider


class _RispostaFinta:
    def __init__(self, status_code: int = 200, content: bytes = b"audio") -> None:
        self.status_code = status_code
        self.content = content
        self.text = ""


class _SessioneFinta:
    """Registra le richieste e quante volte e' stata costruita."""

    costruite = 0

    def __init__(self) -> None:
        type(self).costruite += 1
        self.richieste: list[dict] = []
        self.adapters_montati: list[tuple] = []
        self.chiusa = False
        self.cookies = _CookieJarFinto()
        self.risposta = _RispostaFinta()

    def mount(self, prefisso, adapter):  # noqa: ANN001
        self.adapters_montati.append((prefisso, adapter))

    def post(self, url, **kwargs):  # noqa: ANN001
        self.richieste.append({"url": url, **kwargs})
        return self.risposta

    def close(self) -> None:
        self.chiusa = True


class _CookieJarFinto:
    def __init__(self) -> None:
        self.policy = None

    def set_policy(self, policy) -> None:  # noqa: ANN001
        self.policy = policy


@pytest.fixture(autouse=True)
def _azzera_contatore():
    _SessioneFinta.costruite = 0
    yield


@pytest.fixture
def provider(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "plugins.tts.azure.provider.requests.Session", _SessioneFinta, raising=True
    )
    monkeypatch.setattr(
        "plugins.tts.azure.provider.AzureSpeechTTSProvider._credentials",
        lambda self: ("chiave-finta", "northeurope"),
        raising=True,
    )
    p = AzureSpeechTTSProvider()
    p._uscita = tmp_path
    return p


def _sintetizza(provider, tmp_path, testo="Ciao."):
    uscita = tmp_path / ("%s.mp3" % abs(hash(testo)))
    return provider.synthesize(testo, str(uscita))


def test_due_sintesi_riusano_la_stessa_sessione(provider, tmp_path):
    """Il cuore dell'intervento: una sola sessione per piu' sintesi."""
    _sintetizza(provider, tmp_path, "Prima frase.")
    _sintetizza(provider, tmp_path, "Seconda frase.")

    assert _SessioneFinta.costruite == 1, (
        "La sessione e' stata ricostruita: la connessione non viene riusata e "
        "si ripaga l'handshake TLS (~117 ms) a ogni frase."
    )
    assert len(provider._sessione().richieste) == 2


def test_la_sessione_nasce_una_volta_sola_anche_sotto_concorrenza(provider, tmp_path):
    """Due anziani che parlano insieme non devono costruire due sessioni.

    Il provider e' un singleton di processo (`tts_registry` lo registra una
    volta sola), quindi `synthesize()` puo' entrare da piu' thread insieme. Una
    creazione non protetta lascerebbe due sessioni, e una delle due verrebbe
    buttata via a ogni giro.
    """
    pronti = threading.Barrier(8)
    errori: list[BaseException] = []

    def lavora(i: int) -> None:
        try:
            pronti.wait(timeout=5)
            _sintetizza(provider, tmp_path, "Frase %d." % i)
        except BaseException as exc:  # noqa: BLE001
            errori.append(exc)

    thread = [threading.Thread(target=lavora, args=(i,)) for i in range(8)]
    for t in thread:
        t.start()
    for t in thread:
        t.join(timeout=10)

    assert not errori, "errori nei thread: %r" % errori
    assert _SessioneFinta.costruite == 1, (
        "Sotto concorrenza sono state costruite %d sessioni invece di una."
        % _SessioneFinta.costruite
    )


def test_la_sessione_non_conserva_cookie(provider, tmp_path):
    """La sessione e' condivisa fra famiglie diverse: non deve portarsi dietro
    stato da una richiesta all'altra.

    Una connessione riusata non trasporta ne' voce ne' parametri — quelli stanno
    nel corpo — ma un cookie messo da Azure verrebbe rispedito alla richiesta
    successiva, che e' di un altro anziano. `audio_pipeline.py` documenta il
    guasto peggiore che questo percorso possa produrre: «una frase sbagliata
    nella voce di qualcun altro». Qui si chiude la porta prima di aprirla.
    """
    _sintetizza(provider, tmp_path)
    sessione = provider._sessione()
    assert sessione.cookies.policy is not None, (
        "Nessuna policy sui cookie: la sessione condivisa puo' accumulare stato "
        "fra richieste di famiglie diverse."
    )
    consentiti = sessione.cookies.policy.allowed_domains()
    assert list(consentiti or []) == [], (
        "La policy non blocca tutti i domini: qualche cookie puo' ancora "
        "sopravvivere alla richiesta e partire con quella successiva."
    )


def test_il_pool_e_dimensionato_di_proposito(provider, tmp_path):
    """Il default di requests e' 10 connessioni per host.

    Con una sessione unica di processo quel numero diventa un limite di
    concorrenza, quindi va scelto invece che ereditato.
    """
    _sintetizza(provider, tmp_path)
    montati = provider._sessione().adapters_montati
    assert montati, "Nessun adapter montato: il pool e' quello di default."
    prefissi = {p for p, _ in montati}
    assert "https://" in prefissi


def test_la_richiesta_e_rimasta_quella_di_prima(provider, tmp_path):
    """L'intervento e' solo sul trasporto: url, header e corpo non cambiano."""
    _sintetizza(provider, tmp_path, "Ciao.")
    richiesta = provider._sessione().richieste[0]

    assert richiesta["url"] == (
        "https://northeurope.tts.speech.microsoft.com/cognitiveservices/v1"
    )
    headers = richiesta["headers"]
    assert headers["Ocp-Apim-Subscription-Key"] == "chiave-finta"
    assert headers["Content-Type"] == "application/ssml+xml"
    assert headers["User-Agent"] == "stories-mvp"
    assert b"<speak" in richiesta["data"]
    assert richiesta["timeout"] > 0


def test_un_errore_http_resta_un_errore(provider, tmp_path):
    """La sessione non deve ingoiare i fallimenti."""
    provider._sessione().risposta = _RispostaFinta(status_code=401, content=b"")
    with pytest.raises(RuntimeError, match="401"):
        _sintetizza(provider, tmp_path)


def test_la_sessione_vera_e_costruita_come_dichiarato(monkeypatch):
    """Gli altri test usano una sessione finta, quindi non eseguono mai
    `_crea_sessione()`. Questo la costruisce per davvero e guarda cosa ne esce:
    senza, il pool e la policy potrebbero essere sbagliati e nessun test se ne
    accorgerebbe.

    Non fa nessuna richiesta di rete: costruisce l'oggetto e lo ispeziona.
    """
    from plugins.tts.azure import provider as modulo

    p = modulo.AzureSpeechTTSProvider()
    sessione = p._sessione()
    try:
        assert isinstance(sessione, modulo.requests.Session)

        # la seconda chiamata NON ricostruisce
        assert p._sessione() is sessione

        adapter = sessione.get_adapter("https://northeurope.tts.speech.microsoft.com/")
        assert adapter._pool_maxsize == modulo._POOL_MAXSIZE, (
            "Il pool e' tornato al default di requests (10): con una sessione "
            "unica di processo quel numero e' il tetto di sintesi concorrenti."
        )
        assert adapter._pool_connections == modulo._POOL_CONNESSIONI

        # `_policy` e non `policy`: il jar di requests non espone un getter
        # pubblico. Qui si verifica il nostro cablaggio, non l'API di requests.
        assert sessione.cookies._policy is modulo._POLITICA_SENZA_COOKIE
        assert list(modulo._POLITICA_SENZA_COOKIE.allowed_domains() or []) == []
    finally:
        sessione.close()
