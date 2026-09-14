import pretty_midi
import soundfile as sf


def export_events_to_midi(
    eventos_midi, caminho_arquivo="output.mid", program_instrument=0
):
    """Converte a lista de dicionários de eventos em um arquivo MIDI (.mid).

    Args:
        eventos_midi (list[dict]): Lista com note, start, duration e velocity.
        caminho_arquivo (str): Caminho de saída do arquivo MIDI.
        program_instrument (int): Programa General MIDI (0 = Acoustic Grand
        Piano).

    Returns:
        pretty_midi.PrettyMIDI: Objeto MIDI gerado.
    """
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=program_instrument)

    for ev in eventos_midi:
        nota = pretty_midi.Note(
            velocity=int(ev["velocity"]),
            pitch=int(ev["note"]),
            start=float(ev["start"]),
            end=float(ev["start"] + ev["duration"]),
        )
        piano.notes.append(nota)

    pm.instruments.append(piano)
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
    audio_data = pm.fluidsynth(fs=sr, soundfont_path=soundfont_path)
    sf.write(caminho_wav, audio_data, sr)
    return audio_data