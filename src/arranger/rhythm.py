import numpy as np


def gerar_bloco_sustentado(t_start, duracao_seg, nota_baixo, voicing_md, bpm=120):
    """Estado 0: Baixa densidade.

    Notas longas dão espaço à voz. O baixo sustenta o segmento inteiro; a
    mão direita ataca a cada 2 tempos dentro do segmento 
    """
    seconds_per_beat = 60.0 / bpm
    n_beats = max(1, round(duracao_seg / seconds_per_beat))

    eventos = []

    # Mão Esquerda: Baixo sustentado por quase todo o segmento
    eventos.append({
        "note": int(nota_baixo),
        "start": round(t_start, 4),
        "duration": round(duracao_seg * 0.95, 4),
        "velocity": 70,
        "channel": 0,
    })

    # Mão Direita: Acordes a cada 2 tempos dentro do segmento
    for beat in range(0, n_beats, 2):
        t_inicio = t_start + beat * seconds_per_beat
        tempo_restante = (t_start + duracao_seg) - t_inicio
        duracao_evento = min(seconds_per_beat * 1.8, tempo_restante)
        for note in voicing_md:
            eventos.append({
                "note": int(note),
                "start": round(t_inicio, 4),
                "duration": round(max(duracao_evento, 0.05), 4),
                "velocity": 65,
                "channel": 0,
            })

    return eventos


def gerar_pulso_colcheias(t_start, duracao_seg, nota_baixo, voicing_md, bpm=120):
    """Estado 1: Densidade média.

    Mão direita pulsa o acorde em colcheias ao longo de todo o segmento.
    """
    seconds_per_beat = 60.0 / bpm
    dt_8th = seconds_per_beat / 2
    n_beats = max(1, round(duracao_seg / seconds_per_beat))

    eventos = []

    # Baixo a cada 2 tempos dentro do segmento
    for beat in range(0, n_beats, 2):
        t_inicio = t_start + beat * seconds_per_beat
        tempo_restante = (t_start + duracao_seg) - t_inicio
        eventos.append({
            "note": int(nota_baixo),
            "start": round(t_inicio, 4),
            "duration": round(max(min(seconds_per_beat * 1.8, tempo_restante), 0.05), 4),
            "velocity": 80,
            "channel": 0,
        })

    # Mão direita em colcheias cobrindo o segmento inteiro
    num_colcheias = max(1, round(duracao_seg / dt_8th))
    for colcheia in range(num_colcheias):
        t_inicio = t_start + colcheia * dt_8th
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


def gerar_arpejo_ascendente(t_start, duracao_seg, nota_baixo, voicing_md, bpm=120):
    """Estado 2: Densidade média/alta.

    Distribui as notas do voicing em arpejo contínuo pelo segmento.
    """
    seconds_per_beat = 60.0 / bpm
    dt_8th = seconds_per_beat / 2

    eventos = []

    # Baixo sustentado no início do segmento
    eventos.append({
        "note": int(nota_baixo),
        "start": round(t_start, 4),
        "duration": round(duracao_seg * 0.9, 4),
        "velocity": 75,
        "channel": 0,
    })

    # Dedilhado da Mão Direita ciclando pelas notas do voicing
    num_notas = len(voicing_md)
    num_colcheias = max(1, round(duracao_seg / dt_8th))
    for colcheia in range(num_colcheias):
        t_inicio = t_start + colcheia * dt_8th
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


def gerar_fill_in_agudo(t_start, duracao_seg, nota_baixo, voicing_md, bpm=120):
    """Estado 3: Alta densidade (Preenchimento).

    Se o segmento tiver pelo menos 2 tempos, divide-o ao meio: primeira
    metade com baixo + acorde simples, segunda metade com fraseado rápido
    em semicolcheias no registro agudo. 
    """
    seconds_per_beat = 60.0 / bpm
    dt_16th = seconds_per_beat / 4
    n_beats = duracao_seg / seconds_per_beat

    eventos = []
    voicing_agudo = [n + 12 for n in voicing_md]

    if n_beats >= 2:
        metade_seg = duracao_seg / 2

        # Mão Esquerda cobrindo a primeira metade do segmento
        eventos.append({
            "note": int(nota_baixo),
            "start": round(t_start, 4),
            "duration": round(metade_seg, 4),
            "velocity": 85,
            "channel": 0,
        })

        # Acorde simples na Mão Direita no início da primeira metade e,
        # se houver espaço, também um tempo depois
        pontos_acorde = [0.0] if metade_seg < seconds_per_beat * 1.5 else [0.0, seconds_per_beat]
        for offset in pontos_acorde:
            t_inicio = t_start + offset
            for note in voicing_md:
                eventos.append({
                    "note": int(note),
                    "start": round(t_inicio, 4),
                    "duration": round(seconds_per_beat * 0.8, 4),
                    "velocity": 70,
                    "channel": 0,
                })

        t_fill_start = t_start + metade_seg
        duracao_fill = duracao_seg - metade_seg
    else:
        # Segmento curto demais para dividir: vira fill do início ao fim
        t_fill_start = t_start
        duracao_fill = duracao_seg

    # Fraseamento rápido em semicolcheias (+12 semitons/uma oitava acima)
    num_steps = max(1, round(duracao_fill / dt_16th))
    for step in range(num_steps):
        t_inicio = t_fill_start + step * dt_16th
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


def gerar_padrao_por_estado(estado, t_start, duracao_seg, nota_baixo, voicing_md, bpm=120):
    """Mapeia o estado atual para a função rítmica correspondente, 
       aplicada a um segmento de duração `duracao_seg` (em
       segundos) iniciando em `t_start` (tempo absoluto, em segundos).
    """
    mapa_estados = {
        0: gerar_bloco_sustentado,
        1: gerar_pulso_colcheias,
        2: gerar_arpejo_ascendente,
        3: gerar_fill_in_agudo,
    }

    funcao_geradora = mapa_estados.get(estado, gerar_bloco_sustentado)
    return funcao_geradora(t_start, duracao_seg, nota_baixo, voicing_md, bpm)