import numpy as np

from src.arranger.drums import gerar_faixa_bateria
from src.arranger.rhythm import gerar_padrao_por_estado
from src.arranger.voicing import (
    generate_midi_inversions,
    select_best_voicing,
)
from src.harmony.harmony import NOTE_NAMES, generate_harmonic_field

def _normalizar_para_segmentos(acordes_sugeridos, bpm, beats_per_bar):
    """Aceita tanto o formato legado (list[str], um acorde por compasso)
    quanto o formato de segmentos (list[dict] com "acorde", "tempo_inicio" e
    "tempo_fim" — saída de viterbi.mesclar_em_segmentos) e normaliza para
    uma lista única de segmentos com tempo absoluto em segundos.
    """
    if len(acordes_sugeridos) == 0:
        return []

    # Já está no formato de segmentos
    if isinstance(acordes_sugeridos[0], dict):
        return acordes_sugeridos

    # Formato legado: lista de strings, um acorde por compasso inteiro
    duracao_compasso = (60.0 / bpm) * beats_per_bar
    segmentos = []
    for bar_idx, nome_acorde in enumerate(acordes_sugeridos):
        t_ini = bar_idx * duracao_compasso
        segmentos.append({
            "acorde": nome_acorde,
            "tempo_inicio": t_ini,
            "tempo_fim": t_ini + duracao_compasso,
        })
    return segmentos


def _compassos_cobertos(t_ini, t_fim, duracao_compasso):
    """Índices (0-based) de todos os compassos que se sobrepõem ao
    intervalo [t_ini, t_fim)."""
    primeiro = int(t_ini // duracao_compasso)
    # epsilon evita incluir o compasso seguinte quando t_fim cai exatamente
    # na borda de um compasso
    ultimo = int(max(t_fim - 1e-9, t_ini) // duracao_compasso)
    return list(range(primeiro, max(primeiro, ultimo) + 1))


def arrange_chord_sequence(
    acordes_sugeridos,
    tom_raiz,
    atividade_vocal=None,
    bpm=120,
    beats_per_bar=4,
    limite_fill_sec=2.0,
    incluir_bateria=True,
    num_compassos=None,
):
    """Gera o arranjo MIDI por regras determinísticas baseadas na presença da voz.

    Args:
        acordes_sugeridos: Aceita dois formatos:
            - list[str]: um acorde por compasso inteiro
            - list[dict]: segmentos de duração variável, cada um com as
              chaves "acorde", "tempo_inicio" e "tempo_fim" (em segundos).
        tom_raiz (str): Tom da música.
        atividade_vocal (list[bool], optional): Flag de voz ativa por
            compasso.
        bpm (int): Tempos por minuto.
        beats_per_bar (int): Batidas por compasso.
        limite_fill_sec (float): Tempo máximo (em segundos) de permanência no
          Fill-in durante pausas longas da voz.
        incluir_bateria (bool): Se True, soma um padrão fixo de bateria
          pop/rock à faixa gerada.
        num_compassos (int, optional): Quantidade de compassos para a faixa
          de bateria. Se None, é inferido a partir do maior "tempo_fim"
          entre os segmentos

    Returns:
        list[dict]: Lista de eventos MIDI do arranjo (teclado + bateria).
    """
    campo_harmonico = generate_harmonic_field(tom_raiz)
    todos_eventos = []
    voicing_anterior = None

    duracao_compasso_seg = (60.0 / bpm) * beats_per_bar
    tempo_silencio_acumulado = 0.0

    segmentos = _normalizar_para_segmentos(acordes_sugeridos, bpm, beats_per_bar)

    for segmento in segmentos:
        nome_acorde = segmento["acorde"]
        t_ini = segmento["tempo_inicio"]
        t_fim = segmento["tempo_fim"]
        duracao_segmento = t_fim - t_ini

        if duracao_segmento <= 0:
            continue

        # 1. Identifica a presença de voz no(s) compasso(s) cobertos pelo segmento
        if atividade_vocal:
            compassos_cobertos = _compassos_cobertos(t_ini, t_fim, duracao_compasso_seg)
            voz_ativa = any(
                atividade_vocal[c] for c in compassos_cobertos if c < len(atividade_vocal)
            )
        else:
            voz_ativa = True

        # 2. Regra de Seleção de Estado Rítmico com Limite Temporal
        if voz_ativa:
            tempo_silencio_acumulado = 0.0
            # Voz ativa: mantém o acompanhamento suave (Estado 0: Bloco Sustentado)
            estado_atual = 0
        else:
            tempo_silencio_acumulado += duracao_segmento

            # Se o silêncio acumulado está dentro do limite: aciona a resposta rápida
            if tempo_silencio_acumulado <= limite_fill_sec:
                estado_atual = 3  # Estado 3: FILL_IN_AGUDO (destaque imediato)
            else:
                # Silêncio longo: recua para base contínua sem repetição do Fill
                estado_atual = 2  # Estado 2: ARPEJO_ASCENDENTE

        # 3. Processamento Harmônico e Voicing
        info_acorde = campo_harmonico[nome_acorde]
        notas_indices = [NOTE_NAMES.index(n) for n in info_acorde["notas"]]
        nota_baixo = notas_indices[0] + 36

        inversoes = generate_midi_inversions(notas_indices)
        melhor_voicing = select_best_voicing(voicing_anterior, inversoes)
        voicing_anterior = melhor_voicing

        # 4. Geração do Padrão Rítmico do segmento (duração arbitrária)
        eventos_segmento = gerar_padrao_por_estado(
            estado=estado_atual,
            t_start=t_ini,
            duracao_seg=duracao_segmento,
            nota_baixo=nota_baixo,
            voicing_md=melhor_voicing,
            bpm=bpm,
        )

        todos_eventos.extend(eventos_segmento)

    # 5. Faixa de bateria — padrão fixo por compasso inteiro, independente
    #    da atividade vocal e dos segmentos de acorde
    if incluir_bateria:
        if num_compassos is None:
            if segmentos:
                ultimo_tempo_fim = max(s["tempo_fim"] for s in segmentos)
                num_compassos = int(np.ceil(ultimo_tempo_fim / duracao_compasso_seg))
            else:
                num_compassos = 0

        eventos_bateria = gerar_faixa_bateria(
            num_compassos=num_compassos,
            bpm=bpm,
            beats_per_bar=beats_per_bar,
        )
        todos_eventos.extend(eventos_bateria)

    return todos_eventos