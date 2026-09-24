import numpy as np

def gerar_bloco_sustentado(
    bar_idx, nota_baixo, voicing_md, bpm=120, beats_per_bar=4
):
    """Estado 0: Baixa densidade.

    Notas longas nos tempos 1 e 3 para dar espaço à voz.
    """
    seconds_per_beat = 60.0 / bpm
    bar_start = bar_idx * beats_per_bar * seconds_per_beat
    duracao_compasso = beats_per_bar * seconds_per_beat

    eventos = []

    # Mão Esquerda: Baixo sustentado por quase todo o compasso
    eventos.append({
        "note": int(nota_baixo),
        "start": round(bar_start, 4),
        "duration": round(duracao_compasso * 0.95, 4),
        "velocity": 70,
        "channel": 0,
    })

    # Mão Direita: Acordes no tempo 1 e tempo 3
    for beat in [0, 2]:
        t_inicio = bar_start + (beat * seconds_per_beat)
        for note in voicing_md:
            eventos.append({
                "note": int(note),
                "start": round(t_inicio, 4),
                "duration": round(seconds_per_beat * 1.8, 4),
                "velocity": 65,
                "channel": 0,
            })

    return eventos


def gerar_pulso_colcheias(
    bar_idx, nota_baixo, voicing_md, bpm=120, beats_per_bar=4
):
    """Estado 1: Densidade média.

    Mão direita pulsa o acorde em colcheias (8 vezes por compasso).
    """
    seconds_per_beat = 60.0 / bpm
    bar_start = bar_idx * beats_per_bar * seconds_per_beat
    dt_8th = seconds_per_beat / 2

    eventos = []

    # Baixo nos tempos 1 e 3
    for beat in [0, 2]:
        eventos.append({
            "note": int(nota_baixo),
            "start": round(bar_start + (beat * seconds_per_beat), 4),
            "duration": round(seconds_per_beat * 1.8, 4),
            "velocity": 80,
            "channel": 0,
        })

    # Mão direita em colcheias
    for colcheia in range(beats_per_bar * 2):
        t_inicio = bar_start + (colcheia * dt_8th)
        # Dinâmica levemente acentuada nas batidas principais
        vel = 75 if colcheia % 2 == 0 else 60

        for note in voicing_md:
            eventos.append({
                "note": int(note),
                "start": round(t_inicio, 4),
                "duration": round(dt_8th * 0.85, 4),
                "velocity": vel,
                "channel": 0,
            })

    return eventos


def gerar_arpejo_ascendente(
    bar_idx, nota_baixo, voicing_md, bpm=120, beats_per_bar=4
):
    """Estado 2: Densidade média/alta.

    Distribui as notas do voicing em arpejo contínuo.
    """
    seconds_per_beat = 60.0 / bpm
    bar_start = bar_idx * beats_per_bar * seconds_per_beat
    dt_8th = seconds_per_beat / 2

    eventos = []

    # Baixo sustentado no tempo 1
    eventos.append({
        "note": int(nota_baixo),
        "start": round(bar_start, 4),
        "duration": round(beats_per_bar * seconds_per_beat * 0.9, 4),
        "velocity": 75,
        "channel": 0,
    })

    # Dedilhado da Mão Direita ciclando pelas notas do voicing
    num_notas = len(voicing_md)
    for colcheia in range(beats_per_bar * 2):
        t_inicio = bar_start + (colcheia * dt_8th)
        idx_nota = colcheia % num_notas
        nota = voicing_md[idx_nota]

        eventos.append({
            "note": int(nota),
            "start": round(t_inicio, 4),
            "duration": round(dt_8th * 0.9, 4),
            "velocity": 70,
            "channel": 0,
        })

    return eventos


def gerar_fill_in_agudo(
    bar_idx, nota_baixo, voicing_md, bpm=120, beats_per_bar=4
):
    """Estado 3: Alta densidade (Preenchimento).

    Executa um fraseado rápido no registro agudo nos tempos 3 e 4.
    """
    seconds_per_beat = 60.0 / bpm
    bar_start = bar_idx * beats_per_bar * seconds_per_beat
    dt_16th = seconds_per_beat / 4

    eventos = []

    # Mão Esquerda no tempo 1
    eventos.append({
        "note": int(nota_baixo),
        "start": round(bar_start, 4),
        "duration": round(seconds_per_beat * 2, 4),
        "velocity": 85,
        "channel": 0,
    })

    # Acorde simples na Mão Direita nos tempos 1 e 2
    for beat in [0, 1]:
        t_inicio = bar_start + (beat * seconds_per_beat)
        for note in voicing_md:
            eventos.append({
                "note": int(note),
                "start": round(t_inicio, 4),
                "duration": round(seconds_per_beat * 0.8, 4),
                "velocity": 70,
                "channel": 0,
            })

    # Fraseamento rápido em semicolcheias nos tempos 3 e 4 (+12 semitons/uma oitava acima)
    voicing_agudo = [n + 12 for n in voicing_md]
    t_fill_start = bar_start + (2 * seconds_per_beat)

    for step in range(8):
        t_inicio = t_fill_start + (step * dt_16th)
        idx_nota = step % len(voicing_agudo)
        nota = voicing_agudo[idx_nota]

        eventos.append({
            "note": int(nota),
            "start": round(t_inicio, 4),
            "duration": round(dt_16th * 0.85, 4),
            "velocity": 80,
            "channel": 0,
        })

    return eventos


def gerar_padrao_por_estado(
    estado, bar_idx, nota_baixo, voicing_md, bpm=120, beats_per_bar=4
):
    """Mapeia o estado retornado pela Cadeia de Markov para a função rítmica correspondente."""
    mapa_estados = {
        0: gerar_bloco_sustentado,
        1: gerar_pulso_colcheias,
        2: gerar_arpejo_ascendente,
        3: gerar_fill_in_agudo,
    }

    funcao_geradora = mapa_estados.get(estado, gerar_bloco_sustentado)
    return funcao_geradora(bar_idx, nota_baixo, voicing_md, bpm, beats_per_bar)