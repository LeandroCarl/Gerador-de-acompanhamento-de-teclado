from src.harmony.harmony import *

CUSTO_FUNCAO = {
    ("T", "T"): 0.4, ("T", "SD"): 0.2, ("T", "D"): 0.5,
    ("SD", "T"): 0.3, ("SD", "SD"): 0.6, ("SD", "D"): 0.1,
    ("D", "T"): 0.0, ("D", "SD"): 1.6, ("D", "D"): 0.7,
}


def _custo_emissao(chroma_medio, info, indices_escala):

    custo = 0.0
    notas_acorde_indices = [NOTE_NAMES.index(n) for n in info["notas"]]
    acorde_fora_da_escala = not set(notas_acorde_indices).issubset(
        indices_escala
    )

    # Penalidade fixa caso a tríade do próprio acorde não pertença à escala
    if acorde_fora_da_escala:
        custo += 1.5

    for idx_nota, energia in enumerate(chroma_medio):
        if energia <= 0.01:
            continue

        nota_nome = NOTE_NAMES[idx_nota]

        if nota_nome in info["notas"]:
            # Nível 1: Nota pertence ao acorde
            posicao = info["notas"].index(nota_nome)
            peso_base = 0.5 if posicao == 0 else (1.0 if posicao == 1 else 1.5)
            custo += peso_base * energia

        elif idx_nota in indices_escala:
            # Nível 2: Nota fora do acorde, mas pertence à escala raiz (melodia vocal)
            custo += 3.5 * energia

        else:
            # Nível 3: Nota totalmente fora do tom (ruído / nota cromática)
            custo += 12.0 * energia

    return custo


def mesclar_em_segmentos(caminho, blocos):
    """Colapsa o caminho decodificado (um acorde por bloco da grade) em
    segmentos de duração variável, mesclando blocos consecutivos com o
    mesmo acorde.

    Args:
        caminho (list[str]): Sequência de acordes, um por bloco, na ordem
            de `blocos` (saída do traceback do Viterbi).
        blocos (list[tuple[DataFrame, np.ndarray]]): Lista de
            (bloco_df, chroma_medio) na mesma ordem/tamanho de `caminho`.

    Returns:
        list[dict]: Lista de segmentos com "acorde", "tempo_inicio",
        "tempo_fim" e "num_blocos" (quantos blocos da grade original foram
        mesclados nesse segmento — útil como proxy de duração relativa).
    """
    segmentos = []
    for (bloco_df, _), nome in zip(blocos, caminho):
        t_ini = float(bloco_df["tempo_inicio"].iloc[0])
        t_fim = float(bloco_df["tempo_fim"].iloc[-1])

        if segmentos and segmentos[-1]["acorde"] == nome:
            segmentos[-1]["tempo_fim"] = t_fim
            segmentos[-1]["num_blocos"] += 1
        else:
            segmentos.append({
                "acorde": nome,
                "tempo_inicio": t_ini,
                "tempo_fim": t_fim,
                "num_blocos": 1,
            })

    return segmentos


def suggest_chords_viterbi(
    df_compassos,
    tom_raiz,
    tamanho_bloco=1,
    peso_transicao=1.2,
    permitir_diminutos=False,
    custo_repouso_final=0.0,
    custo_troca_fora_tempo_forte=0.0,
):
    """Decodifica a sequência de acordes via Viterbi.

    Args:
        df_compassos (DataFrame): saída de extract_chroma.
        tamanho_bloco (int): quantas linhas de df_compassos formam um
            "bloco" de decisão do Viterbi.
        custo_troca_fora_tempo_forte (float): pedágio adicional aplicado
            quando o Viterbi troca de acorde (a != b) ENTRANDO em uma
            unidade cuja "posicao_no_compasso" é diferente de 0 (ou seja,
            fora do início do compasso).

    Returns:
        tuple: (acordes_finais, caminho, funcoes, segmentos)
            - acordes_finais: um acorde por linha original de df_compassos
            - caminho: um acorde por bloco decodificado.
            - funcoes: mapa acorde -> função tonal (T/SD/D).
            - segmentos: lista de dicts (acorde, tempo_inicio, tempo_fim,
              num_blocos) com blocos consecutivos iguais já mesclados
    """

    campo = generate_harmonic_field(tom_raiz)
    _, indices_escala = get_scale_notes(tom_raiz)
    _, modo = extract_tone_and_mode(tom_raiz)
    funcao_map = FUNCAO_MENOR if modo == "menor" else FUNCAO_MAIOR
    funcoes = {nome: funcao_map.get(info["grau"], "T") for nome, info in campo.items()}

    nomes_acordes = list(campo.keys())
    if not permitir_diminutos:
        nomes_acordes = [n for n in nomes_acordes if "dim" not in n]

    tem_posicao_metrica = "posicao_no_compasso" in df_compassos.columns

    # agrupa em blocos e pré-calcula o chroma médio + posição métrica de cada um
    blocos = []
    for i in range(0, len(df_compassos), tamanho_bloco):
        bloco = df_compassos.iloc[i:i + tamanho_bloco]
        chroma_matrix = np.array(bloco["chroma"].tolist())
        chroma_medio = np.mean(chroma_matrix, axis=0)
        # posição métrica do bloco = posição da primeira linha que o compõe
        posicao_metrica = (
            int(bloco["posicao_no_compasso"].iloc[0]) if tem_posicao_metrica else 0
        )
        blocos.append((bloco, chroma_medio, posicao_metrica))
    n = len(blocos)

    def custo_transicao(a, b, posicao_destino):
        base = CUSTO_FUNCAO.get((funcoes[a], funcoes[b]), 0.5)
        if a == b:
            base += 0.3  # leve custo extra por repetir o mesmo acorde literal
        elif posicao_destino != 0:
            # trocar de acorde fora do início do compasso é mais caro —
            # só compensa se a emissão (chroma) apoiar fortemente a troca
            base += custo_troca_fora_tempo_forte
        return base * peso_transicao

    # --- Viterbi ---
    dp = [dict() for _ in range(n)]
    bt = [dict() for _ in range(n)]

    for nome in nomes_acordes:
        dp[0][nome] = _custo_emissao(blocos[0][1], campo[nome], indices_escala)
        bt[0][nome] = None

    for t in range(1, n):
        chroma_medio = blocos[t][1]
        posicao_destino = blocos[t][2]
        for nome in nomes_acordes:
            e = _custo_emissao(chroma_medio, campo[nome], indices_escala)
            melhor_custo, melhor_prev = None, None
            for prev in nomes_acordes:
                total = dp[t - 1][prev] + custo_transicao(prev, nome, posicao_destino) + e
                if melhor_custo is None or total < melhor_custo:
                    melhor_custo, melhor_prev = total, prev
            dp[t][nome] = melhor_custo
            bt[t][nome] = melhor_prev

    # traceback do caminho de menor custo total — penalizando terminar fora da Tônica
    custo_final = {
        nome: dp[-1][nome] + (0.0 if funcoes[nome] == "T" else custo_repouso_final)
        for nome in nomes_acordes
    }
    ultimo = min(custo_final, key=custo_final.get)
    caminho = [ultimo]
    for t in range(n - 1, 0, -1):
        caminho.append(bt[t][caminho[-1]])
    caminho.reverse()

    acordes_finais = []
    for (bloco, _, _), nome in zip(blocos, caminho):
        acordes_finais.extend([nome] * len(bloco))

    blocos_para_merge = [(bloco_df, chroma) for (bloco_df, chroma, _) in blocos]
    segmentos = mesclar_em_segmentos(caminho, blocos_para_merge)

    return acordes_finais, caminho, funcoes, segmentos