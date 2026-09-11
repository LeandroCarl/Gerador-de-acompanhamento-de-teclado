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


def suggest_chords_viterbi(df_compassos, tom_raiz, tamanho_bloco=1, peso_transicao=1.2,
                             permitir_diminutos=False, custo_repouso_final=0.0):
    
    campo = generate_harmonic_field(tom_raiz)
    _, indices_escala = get_scale_notes(tom_raiz)
    _, modo = extract_tone_and_mode(tom_raiz)
    funcao_map = FUNCAO_MENOR if modo == "menor" else FUNCAO_MAIOR
    funcoes = {nome: funcao_map.get(info["grau"], "T") for nome, info in campo.items()}

    nomes_acordes = list(campo.keys())
    if not permitir_diminutos:
        nomes_acordes = [n for n in nomes_acordes if "dim" not in n]

    # agrupa em blocos e pré-calcula o chroma médio de cada um
    blocos = []
    for i in range(0, len(df_compassos), tamanho_bloco):
        bloco = df_compassos.iloc[i:i + tamanho_bloco]
        chroma_matrix = np.array(bloco["chroma"].tolist())
        chroma_medio = np.mean(chroma_matrix, axis=0)
        blocos.append((bloco, chroma_medio))
    n = len(blocos)

    def custo_transicao(a, b):
        base = CUSTO_FUNCAO.get((funcoes[a], funcoes[b]), 0.5)
        if a == b:
            base += 0.3  # leve custo extra por repetir o mesmo acorde literal
        return base * peso_transicao

    # --- Viterbi ---
    dp = [dict() for _ in range(n)]
    bt = [dict() for _ in range(n)]

    for nome in nomes_acordes:
        dp[0][nome] = _custo_emissao(blocos[0][1], campo[nome], indices_escala)
        bt[0][nome] = None

    for t in range(1, n):
        chroma_medio = blocos[t][1]
        for nome in nomes_acordes:
            e = _custo_emissao(chroma_medio, campo[nome], indices_escala)
            melhor_custo, melhor_prev = None, None
            for prev in nomes_acordes:
                total = dp[t - 1][prev] + custo_transicao(prev, nome) + e
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
    for (bloco, _), nome in zip(blocos, caminho):
        acordes_finais.extend([nome] * len(bloco))

    return acordes_finais, caminho, funcoes