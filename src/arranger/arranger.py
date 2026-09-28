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


def _indice_unidade_vocal(t_cursor, duracao_subbloco, t_anacruse, duracao_unidade):
    """Índice (0-based) da unidade de atividade_vocal correspondente a um
    sub-bloco, calculado a partir do PONTO MÉDIO do sub-bloco (não dos seus
    extremos). 
    """
    meio = t_cursor + (duracao_subbloco / 2.0)
    return int((meio - t_anacruse) // duracao_unidade)


def arrange_chord_sequence(
    acordes_sugeridos,
    tom_raiz,
    atividade_vocal=None,
    bpm=120,
    beats_per_bar=4,
    limite_fill_sec=2.0,
    incluir_bateria=True,
    num_compassos=None,
    unidades_por_compasso_vocal=1,
    offset=0.0,
):
    """Gera o arranjo MIDI por regras determinísticas baseadas na presença da voz.

    Args:
        acordes_sugeridos: Aceita dois formatos:
            - list[str]: um acorde por compasso inteiro.
            - list[dict]: segmentos de duração variável, cada um com as
              chaves "acorde", "tempo_inicio" e "tempo_fim" (em segundos)
        tom_raiz (str): Tom da música.
        atividade_vocal (list[bool], optional): Saída de
            pitch.extract_vocal_activity. A granularidade dessa lista é
            declarada por `unidades_por_compasso_vocal`  
        bpm (int): Tempos por minuto.
        beats_per_bar (int): Batidas por compasso.
        limite_fill_sec (float): Tempo máximo (em segundos) de permanência no
          Fill-in durante pausas longas da voz. 
        incluir_bateria (bool): Se True, soma um padrão fixo de bateria
          pop/rock à faixa gerada. 
        num_compassos (int, optional): Quantidade de compassos para a faixa
          de bateria. 
        unidades_por_compasso_vocal (int): Em quantas partes iguais cada
          compasso foi subdividido ao gerar `atividade_vocal`.
          Default 1 = uma entrada de atividade_vocal por compasso inteiro.
        offset (float): Deslocamento em TEMPOS (batidas) do primeiro tempo
          forte.
    Returns:
        list[dict]: Lista de eventos MIDI do arranjo (teclado + bateria).
    """
    campo_harmonico = generate_harmonic_field(tom_raiz)
    todos_eventos = []
    voicing_anterior = None

    duracao_compasso_seg = (60.0 / bpm) * beats_per_bar
    duracao_unidade_vocal = duracao_compasso_seg / unidades_por_compasso_vocal
    t_anacruse = offset * (60.0 / bpm)
    tempo_silencio_acumulado = 0.0

    segmentos = _normalizar_para_segmentos(acordes_sugeridos, bpm, beats_per_bar)

    for segmento in segmentos:
        nome_acorde = segmento["acorde"]
        t_ini = segmento["tempo_inicio"]
        t_fim = segmento["tempo_fim"]
        duracao_segmento = t_fim - t_ini

        if duracao_segmento <= 0:
            continue

        # 1. Processamento Harmônico e Voicing — uma vez por segmento (mesmo
        #    acorde do início ao fim, mesmo que o segmento seja subdividido
        #    logo abaixo para fins de decisão de estado rítmico)
        info_acorde = campo_harmonico[nome_acorde]
        notas_indices = [NOTE_NAMES.index(n) for n in info_acorde["notas"]]
        nota_baixo = notas_indices[0] + 36

        inversoes = generate_midi_inversions(notas_indices)
        melhor_voicing = select_best_voicing(voicing_anterior, inversoes)
        voicing_anterior = melhor_voicing

        # 2. Subdivide o segmento em sub-blocos do tamanho de UMA unidade de
        #    atividade_vocal (duracao_unidade_vocal) e decide o estado
        #    rítmico separadamente para cada um. 
        t_cursor = t_ini
        while t_cursor < t_fim - 1e-9:
            t_fim_subbloco = min(t_cursor + duracao_unidade_vocal, t_fim)

            # Absorve resíduos minúsculos de arredondamento no sub-bloco
            # atual 
            if 0.0 < (t_fim - t_fim_subbloco) < 0.01:
                t_fim_subbloco = t_fim

            duracao_subbloco = t_fim_subbloco - t_cursor

            # 2a. Identifica a presença de voz na unidade de atividade_vocal
            #    correspondente a este sub-bloco (índice único via ponto
            #    médio — ver _indice_unidade_vocal)
            if atividade_vocal:
                idx_unidade = _indice_unidade_vocal(
                    t_cursor, duracao_subbloco, t_anacruse, duracao_unidade_vocal
                )
                voz_ativa = (
                    atividade_vocal[idx_unidade]
                    if 0 <= idx_unidade < len(atividade_vocal)
                    else False
                )
            else:
                voz_ativa = True

            # 2b. Regra de Seleção de Estado Rítmico com Limite Temporal
            if voz_ativa:
                tempo_silencio_acumulado = 0.0
                # Voz ativa: mantém o acompanhamento suave (Estado 0: Bloco Sustentado)
                estado_atual = 0
            else:
                tempo_silencio_acumulado += duracao_subbloco

                # Se o silêncio acumulado está dentro do limite: aciona a resposta rápida
                if tempo_silencio_acumulado <= limite_fill_sec:
                    estado_atual = 3  # Estado 3: FILL_IN_AGUDO (destaque imediato)
                else:
                    # Silêncio longo: recua para base contínua sem repetição do Fill
                    estado_atual = 2  # Estado 2: ARPEJO_ASCENDENTE

            # 2c. Geração do Padrão Rítmico do sub-bloco (duração arbitrária)
            eventos_subbloco = gerar_padrao_por_estado(
                estado=estado_atual,
                t_start=t_cursor,
                duracao_seg=duracao_subbloco,
                nota_baixo=nota_baixo,
                voicing_md=melhor_voicing,
                bpm=bpm,
            )

            todos_eventos.extend(eventos_subbloco)
            t_cursor = t_fim_subbloco

    # 3. Faixa de bateria — padrão fixo por compasso inteiro, independente
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

def humanizar_eventos_midi(
    eventos_midi, vel_std=5, timing_std=0.010, roll_chord_ms=0.012
):
    """Aplica variações humanas de dinâmica e tempo aos eventos MIDI.

    Args:
        eventos_midi (list[dict]): Lista de eventos gerados.
        vel_std (int): Desvio padrão da intensidade (Velocity).
        timing_std (float): Desvio temporal aleatório em segundos (ex: 0.010s =
          10ms).
        roll_chord_ms (float): Micro-atraso entre as notas de um mesmo acorde
          (dedilhado humano).

    Returns:
        list[dict]: Lista de eventos humanizados.
    """
    eventos_humanizados = []

    # Agrupa notas por tempo de início exato para identificar acordes da Mão Direita
    tempo_grupos = {}
    for ev in eventos_midi:
        t = ev["start"]
        tempo_grupos.setdefault(t, []).append(ev.copy())

    for t_original, grupo in tempo_grupos.items():
        # Separa a nota do baixo (mão esquerda) das notas do acorde (mão direita)
        notas_ordenadas = sorted(grupo, key=lambda x: x["note"])

        for idx, ev in enumerate(notas_ordenadas):
            # 1. Micro-timing jitter (atraso/antecipação aleatória de ~5 a 10ms)
            delta_tempo = np.random.normal(0, timing_std)

            # 2. Staggering/Roll: em acordes, dedos diferentes tocam em instantes levemente diferentes
            # A nota mais grave do acorde soa ligeiramente antes das mais agudas
            atraso_dedo = idx * roll_chord_ms if len(grupo) > 1 else 0.0

            ev["start"] = max(0.0, round(ev["start"] + delta_tempo + atraso_dedo, 4))

            # 3. Variação de Velocity (dinâmica da força do dedo)
            delta_vel = int(np.random.normal(0, vel_std))
            ev["velocity"] = int(np.clip(ev["velocity"] + delta_vel, 30, 127))

            eventos_humanizados.append(ev)

    # Reordena a lista por tempo de início ajustado
    return sorted(eventos_humanizados, key=lambda x: x["start"])