from src.arranger.drums import gerar_faixa_bateria
from src.arranger.rhythm import gerar_padrao_por_estado
from src.arranger.voicing import (
    generate_midi_inversions,
    select_best_voicing,
)
from src.harmony.harmony import NOTE_NAMES, generate_harmonic_field


def arrange_chord_sequence(
    acordes_sugeridos,
    tom_raiz,
    atividade_vocal=None,
    bpm=120,
    beats_per_bar=4,
    limite_fill_sec=2.0,
    incluir_bateria=True,
):
    """Gera o arranjo MIDI por regras determinísticas baseadas na presença da voz.

    Args:
        acordes_sugeridos (list[str]): Sequência de acordes do Viterbi.
        tom_raiz (str): Tom da música.
        atividade_vocal (list[bool], optional): Flag de voz ativa por compasso.
        bpm (int): Tempos por minuto.
        beats_per_bar (int): Batidas por compasso.
        limite_fill_sec (float): Tempo máximo (em segundos) de permanência no
          Fill-in durante pausas longas da voz.
        incluir_bateria (bool): Se True, soma um padrão fixo de bateria
          pop/rock (ver drums.py) à faixa gerada. Diferente do teclado, a
          bateria NÃO reage a `atividade_vocal` — toca igual do início ao
          fim; a reatividade à voz fica só por conta do teclado.

    Returns:
        list[dict]: Lista de eventos MIDI do arranjo (teclado + bateria).
    """
    campo_harmonico = generate_harmonic_field(tom_raiz)
    todos_eventos = []
    voicing_anterior = None

    duracao_compasso_seg = (60.0 / bpm) * beats_per_bar
    tempo_silencio_acumulado = 0.0

    for bar_idx, nome_acorde in enumerate(acordes_sugeridos):
        # 1. Identifica a presença de voz no compasso
        voz_ativa = (
            atividade_vocal[bar_idx]
            if (atividade_vocal and bar_idx < len(atividade_vocal))
            else True
        )

        # 2. Regra de Seleção de Estado Rítmico com Limite Temporal
        if voz_ativa:
            tempo_silencio_acumulado = 0.0
            # Voz ativa: mantém o acompanhamento suave (Estado 0: Bloco Sustentado)
            estado_atual = 0
        else:
            tempo_silencio_acumulado += duracao_compasso_seg

            # Se o silêncio está dentro do limite de 2 segundos: aciona a resposta rápida
            if tempo_silencio_acumulado <= limite_fill_sec:
                estado_atual = 3  # Estado 3: FILL_IN_AGUDO (destaque imediato)
            else:
                # Silêncio longo (> 2s): recua para base contínua sem repetição do Fill
                estado_atual = 2  # Estado 2: ARPEJO_ASCENDENTE (ou Estado 1)

        # 3. Processamento Harmônico e Voicing
        info_acorde = campo_harmonico[nome_acorde]
        notas_indices = [NOTE_NAMES.index(n) for n in info_acorde["notas"]]
        nota_baixo = notas_indices[0] + 36

        inversoes = generate_midi_inversions(notas_indices)
        melhor_voicing = select_best_voicing(voicing_anterior, inversoes)
        voicing_anterior = melhor_voicing

        # 4. Geração do Padrão Rítmico do Compasso
        eventos_compasso = gerar_padrao_por_estado(
            estado=estado_atual,
            bar_idx=bar_idx,
            nota_baixo=nota_baixo,
            voicing_md=melhor_voicing,
            bpm=bpm,
            beats_per_bar=beats_per_bar,
        )

        todos_eventos.extend(eventos_compasso)

    # 5. Faixa de bateria — padrão fixo, independente da atividade vocal
    if incluir_bateria:
        eventos_bateria = gerar_faixa_bateria(
            num_compassos=len(acordes_sugeridos),
            bpm=bpm,
            beats_per_bar=beats_per_bar,
        )
        todos_eventos.extend(eventos_bateria)

    return todos_eventos