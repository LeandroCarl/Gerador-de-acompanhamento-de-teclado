"""
Mapeamento General MIDI usado (canal de percussão = 9, zero-based / canal
10 na numeração 1-indexed que a maioria dos DAWs mostra):
    36 = Bumbo (Bass Drum 1)
    38 = Caixa (Acoustic Snare)
    42 = Chimbal fechado (Closed Hi-Hat)
    46 = Chimbal aberto (Open Hi-Hat) — não usado no padrão básico
    49 = Crash 1 — não usado no padrão básico
"""

CANAL_BATERIA = 9  # canal 10 (GM percussion), zero-based

BUMBO = 36
CAIXA = 38
CHIMBAL_FECHADO = 42
CHIMBAL_ABERTO = 46
CRASH = 49


def gerar_batida_pop_compasso(bar_idx, bpm=120, beats_per_bar=4):
    """Um compasso do padrão de bateria pop básico.

    Padrão (compasso 4/4):
        - Bumbo:  tempos 1 e 3
        - Caixa:  tempos 2 e 4
        - Chimbal fechado: colcheias (8 por compasso), levemente
          acentuado nos tempos cheios (1, 2, 3, 4) em relação aos
          contratempos.

    Args:
        bar_idx (int): Índice do compasso (0-based), usado para calcular
            o tempo absoluto de início dos eventos.
        bpm (int): Tempos por minuto.
        beats_per_bar (int): Batidas por compasso (mantido como parâmetro
            por consistência com rhythm.py, mas o padrão abaixo assume 4/4).

    Returns:
        list[dict]: Eventos MIDI do compasso, no mesmo formato usado em
        rhythm.py (note, start, duration, velocity, channel).
    """
    seconds_per_beat = 60.0 / bpm
    bar_start = bar_idx * beats_per_bar * seconds_per_beat
    dt_8th = seconds_per_beat / 2

    eventos = []

    # Bumbo nos tempos 1 e 3
    for beat in [0, 2]:
        eventos.append({
            "note": BUMBO,
            "start": round(bar_start + beat * seconds_per_beat, 4),
            "duration": round(seconds_per_beat * 0.9, 4),
            "velocity": 100,
            "channel": CANAL_BATERIA,
        })

    # Caixa nos tempos 2 e 4
    for beat in [1, 3]:
        eventos.append({
            "note": CAIXA,
            "start": round(bar_start + beat * seconds_per_beat, 4),
            "duration": round(seconds_per_beat * 0.7, 4),
            "velocity": 95,
            "channel": CANAL_BATERIA,
        })

    # Chimbal fechado em colcheias, acentuando os tempos cheios
    for colcheia in range(beats_per_bar * 2):
        t_inicio = bar_start + colcheia * dt_8th
        tempo_cheio = (colcheia % 2 == 0)
        eventos.append({
            "note": CHIMBAL_FECHADO,
            "start": round(t_inicio, 4),
            "duration": round(dt_8th * 0.9, 4),
            "velocity": 80 if tempo_cheio else 60,
            "channel": CANAL_BATERIA,
        })

    return eventos


def gerar_faixa_bateria(num_compassos, bpm=120, beats_per_bar=4):
    """Gera a faixa de bateria completa, compasso a compasso.

    Args:
        num_compassos (int): Quantidade de compassos da música (geralmente
            len(acordes_sugeridos)).
        bpm (int): Tempos por minuto.
        beats_per_bar (int): Batidas por compasso.

    Returns:
        list[dict]: Todos os eventos MIDI de bateria da música.
    """
    eventos = []
    for bar_idx in range(num_compassos):
        eventos.extend(
            gerar_batida_pop_compasso(bar_idx, bpm=bpm, beats_per_bar=beats_per_bar)
        )
    return eventos
