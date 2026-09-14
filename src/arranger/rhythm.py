import numpy as np

def generate_quarter_note_block(
    bar_idx,
    nota_baixo,
    voicing_mao_direita,
    bpm=120,
    beats_per_bar=4,
    gate=0.9,
):
    """Gera eventos MIDI para um compasso no padrão de semínimas em blocos.

    Args:
        bar_idx (int): Índice do compasso (0, 1, 2...).
        nota_baixo (int): Nota MIDI da mão esquerda (ex: 36 para C2).
        voicing_mao_direita (list/array): Notas MIDI da mão direita (ex: [60,
          64, 67]).
        bpm (float): Tempos por minuto.
        beats_per_bar (int): Número de batidas por compasso (padrão: 4).
        gate (float): Fator de articulação (0.9 = 90% da duração da nota, 10% de
          silêncio entre notas).

    Returns:
        list[dict]: Lista de dicionários representando eventos de notas MIDI.
    """
    seconds_per_beat = 60.0 / bpm
    bar_start_time = bar_idx * beats_per_bar * seconds_per_beat
    duracao_nota = seconds_per_beat * gate

    eventos_midi = []

    for beat in range(beats_per_bar):
        tempo_inicio = bar_start_time + (beat * seconds_per_beat)

        # Dinâmica: batidas 1 e 3 têm acento levemente maior (mais forte)
        is_strong_beat = beat in [0, 2]
        vel_lh = 85 if is_strong_beat else 75
        vel_rh = 75 if is_strong_beat else 65

        # Mão Esquerda (Baixo)
        eventos_midi.append({
            "note": int(nota_baixo),
            "start": round(tempo_inicio, 4),
            "duration": round(duracao_nota, 4),
            "velocity": vel_lh,
            "channel": 0,
        })

        # Mão Direita (Bloco de Acordes)
        for note in voicing_mao_direita:
            eventos_midi.append({
                "note": int(note),
                "start": round(tempo_inicio, 4),
                "duration": round(duracao_nota, 4),
                "velocity": vel_rh,
                "channel": 0,
            })

    return eventos_midi