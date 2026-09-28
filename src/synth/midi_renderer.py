import pretty_midi
import soundfile as sf
import numpy as np

CANAL_BATERIA = 9  # precisa bater com o valor usado em drums.py


def export_events_to_midi(
    eventos_midi, caminho_arquivo="output.mid", program_instrument=0
):
    """Converte a lista de dicionários de eventos em um arquivo MIDI (.mid).
    Args:
        eventos_midi (list[dict]): Lista com note, start, duration,
            velocity e channel.
        caminho_arquivo (str): Caminho de saída do arquivo MIDI.
        program_instrument (int): Programa General MIDI para os canais que
            NÃO são bateria (0 = Acoustic Grand Piano).

    Returns:
        pretty_midi.PrettyMIDI: Objeto MIDI gerado.
    """
    pm = pretty_midi.PrettyMIDI()
    # Agrupa eventos por canal, preservando a ordem de chegada
    eventos_por_canal = {}
    for ev in eventos_midi:
        canal = ev.get("channel", 0)
        eventos_por_canal.setdefault(canal, []).append(ev)

    for canal, eventos_canal in eventos_por_canal.items():
        eh_bateria = (canal == CANAL_BATERIA)

        instrumento = pretty_midi.Instrument(
            program=0 if eh_bateria else program_instrument,
            is_drum=eh_bateria,
            name="Bateria" if eh_bateria else "Teclado",
        )

        for ev in eventos_canal:
            nota = pretty_midi.Note(
                velocity=int(ev["velocity"]),
                pitch=int(ev["note"]),
                start=float(ev["start"]),
                end=float(ev["start"] + ev["duration"]),
            )
            instrumento.notes.append(nota)

        pm.instruments.append(instrumento)

    pm.write(caminho_arquivo)
    return pm


def render_audio_wav(
    pm, caminho_wav="output.wav", sr=44100, soundfont_path=None
):
    """Sintetiza o objeto PrettyMIDI em um arquivo de áudio WAV.

    Args:
        pm (pretty_midi.PrettyMIDI): Instância do MIDI criado.
        caminho_wav (str): Caminho para salvar o áudio.
        sr (int): Taxa de amostragem (Hz).
        soundfont_path (str, optional): Caminho de um arquivo SoundFont (.sf2).
          Se None, utiliza o sintetizador padrão de ondas senoidais.

    Returns:
        np.ndarray: Vetor do sinal de áudio gerado.
    """
    audio_data = pm.fluidsynth(fs=sr, sf2_path=soundfont_path)
    sf.write(caminho_wav, audio_data, sr)
    return audio_data

def _envelope_exponencial(n_samples, fs, decaimento=30.0):
    t = np.arange(n_samples) / fs
    return np.exp(-decaimento * t)


def _sintetizar_bumbo(duracao, fs):
    """Tom grave com glide descendente + decaimento rápido — timbre de bumbo."""
    n = int(duracao * fs)
    t = np.arange(n) / fs
    freq_inicial, freq_final = 150.0, 50.0
    freq = freq_final + (freq_inicial - freq_final) * np.exp(-t * 40)
    fase = 2 * np.pi * np.cumsum(freq) / fs
    onda = np.sin(fase)
    onda *= _envelope_exponencial(n, fs, decaimento=18.0)
    return onda


def _sintetizar_ruido(duracao, fs, decaimento=35.0, passa_alta=False):
    """Ruído branco com decaimento — base para caixa e chimbal.

    `passa_alta=True` aplica uma diferenciação simples (passa-alta
    grosseiro) pra dar um timbre mais seco/agudo, característico de
    chimbal; sem isso o ruído soa mais cheio, mais parecido com caixa.
    """
    n = int(duracao * fs)
    ruido = np.random.uniform(-1, 1, n)
    if passa_alta:
        ruido = np.diff(ruido, prepend=0.0)
    ruido *= _envelope_exponencial(n, fs, decaimento=decaimento)
    return ruido


# Nota GM -> gerador de forma de onda (duracao_hit, fs) -> np.ndarray
MAPA_PERCUSSAO = {
    36: lambda dur, fs: _sintetizar_bumbo(dur, fs),                                   # Bumbo
    38: lambda dur, fs: _sintetizar_ruido(dur, fs, decaimento=25.0, passa_alta=False),  # Caixa
    42: lambda dur, fs: _sintetizar_ruido(dur, fs, decaimento=60.0, passa_alta=True),   # Chimbal fechado
    46: lambda dur, fs: _sintetizar_ruido(dur, fs, decaimento=20.0, passa_alta=True),   # Chimbal aberto
    49: lambda dur, fs: _sintetizar_ruido(dur, fs, decaimento=8.0, passa_alta=True),    # Crash
}
_GERADOR_PADRAO = lambda dur, fs: _sintetizar_ruido(dur, fs)  # fallback p/ nota GM não mapeada


def synthesize_drum_instrument(instrumento_bateria, fs=22050, duracao_hit=0.15):
    """Sintetiza um pretty_midi.Instrument de percussão (is_drum=True).

    Args:
        instrumento_bateria (pretty_midi.Instrument): instrumento com
            is_drum=True, cujas notes[].pitch seguem o mapa GM de
            percussão (ver drums.py).
        fs (int): taxa de amostragem.
        duracao_hit (float): duração fixa de cada "batida" sintetizada,
            em segundos — percussão tem ataque/decaimento curto e não
            depende tanto da duration original da nota quanto um
            instrumento melódico dependeria.

    Returns:
        np.ndarray: áudio mono do instrumento de percussão.
    """
    duracao_total = instrumento_bateria.get_end_time() + duracao_hit
    n_amostras = int(duracao_total * fs) + 1
    audio = np.zeros(n_amostras)

    for nota in instrumento_bateria.notes:
        gerador = MAPA_PERCUSSAO.get(nota.pitch, _GERADOR_PADRAO)
        hit = gerador(duracao_hit, fs) * (nota.velocity / 127.0)

        inicio_amostra = int(nota.start * fs)
        fim_amostra = inicio_amostra + len(hit)
        if fim_amostra > len(audio):
            audio = np.pad(audio, (0, fim_amostra - len(audio)))
        audio[inicio_amostra:fim_amostra] += hit

    return audio


def synthesize_with_drums(pm, fs=22050, drum_gain=2.5):
    """Substitui pm.synthesize(fs=fs) quando o PrettyMIDI tem bateria.
    Args:
        drum_gain (float): ganho aplicado só na bateria antes de somar
            com o resto da mixagem.
    """
    total_duration = pm.get_end_time() + 0.5
    n_samples_total = int(total_duration * fs) + 1
    audio_total = np.zeros(n_samples_total)
 
    for instrument in pm.instruments:
        if instrument.is_drum:
            chunk = synthesize_drum_instrument(instrument, fs=fs) * drum_gain
        else:
            chunk = instrument.synthesize(fs=fs, wave=np.sin)
 
        if len(chunk) > len(audio_total):
            audio_total = np.pad(audio_total, (0, len(chunk) - len(audio_total)))
        audio_total[:len(chunk)] += chunk
 
    return audio_total