"""Avaliação da compatibilidade entre a melodia e os acordes sugeridos.

Usa o mesmo agrupamento de chroma e a mesma função de custo de emissão
do decodificador Viterbi. Um alerta de custo é local: não significa, por si
só, que a sequência escolhida pelo Viterbi esteja errada, pois o Viterbi
também considera os custos de transição entre acordes.
"""

import numpy as np
import pandas as pd

from src.harmony.harmony import *
from src.harmony.viterbi import _custo_emissao

from src.harmony.viterbi import (
    _custo_emissao,
    CUSTO_FUNCAO,
)


def avaliar_compatibilidade(
    df_compassos,
    acordes,
    tom_raiz,
    tamanho_bloco=1,
    permitir_diminutos=False,
    limiar_notas_acorde=30.0,
    limiar_notas_cromaticas=20.0,
    tolerancia_custo=0.10,
    voz_ativa=None,
    progressao_original=None,
):
    """Avalia os acordes sugeridos, agrupando o chroma como no Viterbi.

    Args:
        df_compassos: DataFrame com uma linha por unidade e coluna chroma
            contendo vetores de 12 valores.
        acordes: sequência por linha original (acordes_finais) ou por bloco
            (caminho), conforme o retorno de suggest_chords_viterbi.
        tom_raiz: tonalidade, por exemplo "G" ou "Am".
        tamanho_bloco: número de linhas agrupadas por decisão; use o mesmo
            valor empregado na geração dos acordes.
        permitir_diminutos: inclui acordes diminutos nos candidatos.
        limiar_notas_acorde: alerta se energia nas notas do acorde ficar
            abaixo deste percentual.
        limiar_notas_cromaticas: alerta se energia cromática ultrapassar
            este percentual.
        tolerancia_custo: alerta se o custo do acorde escolhido exceder o
            melhor custo local por mais que este valor.
        voz_ativa: sequência opcional de valores booleanos, um por linha de
            df_compassos. Se omitida, usa a coluna df_compassos["voz_ativa"]
            quando existir; se nenhuma fonte existir, a atividade é desconhecida.
        progressao_original: sequência opcional de acordes da cifra original,
            com um acorde por bloco. Seu tamanho deve ser igual ao número de
            blocos avaliados; permite comparar a saída com a música de referência.

    Returns:
        (DataFrame, dict): tabela de avaliação por bloco e resumo agregado.
    """
    if df_compassos is None or df_compassos.empty:
        raise ValueError("df_compassos está vazio.")
    if "chroma" not in df_compassos.columns:
        raise ValueError("df_compassos precisa conter a coluna 'chroma'.")
    if not isinstance(tamanho_bloco, int) or tamanho_bloco < 1:
        raise ValueError("tamanho_bloco deve ser um inteiro maior que zero.")

    # A atividade vocal pode ser fornecida explicitamente ou como coluna do DataFrame.
    if voz_ativa is not None:
        atividade_linhas = list(voz_ativa)
        if len(atividade_linhas) != len(df_compassos):
            raise ValueError(
                "voz_ativa deve ter o mesmo número de elementos que df_compassos."
            )
    elif "voz_ativa" in df_compassos.columns:
        atividade_linhas = df_compassos["voz_ativa"].tolist()
    else:
        atividade_linhas = None

    def _esta_ativa(valor):
        if valor is None or pd.isna(valor):
            return False
        if isinstance(valor, str):
            return valor.strip().lower() in {"true", "1", "sim", "yes", "ativo", "ativa"}
        return bool(valor)

    campo = generate_harmonic_field(tom_raiz)
    _, indices_escala = get_scale_notes(tom_raiz)
    candidatos = list(campo.keys())
    if not permitir_diminutos:
        candidatos = [nome for nome in candidatos if "dim" not in nome]

    blocos = []
    for inicio in range(0, len(df_compassos), tamanho_bloco):
        bloco_df = df_compassos.iloc[inicio:inicio + tamanho_bloco]
        try:
            matriz = np.asarray(bloco_df["chroma"].tolist(), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Chroma inválido no bloco iniciado na linha {inicio}."
            ) from exc

        if matriz.ndim != 2 or matriz.shape[1] != 12:
            raise ValueError("Cada valor de 'chroma' deve conter 12 valores.")
        if not np.isfinite(matriz).all():
            raise ValueError(
                f"Chroma contém valores não finitos no bloco iniciado na linha {inicio}."
            )

        # Igual ao agrupamento usado em suggest_chords_viterbi.
        # Se a atividade vocal estiver disponível, preserva-a no nível do bloco.
        atividade_bloco = None
        if atividade_linhas is not None:
            atividade_bloco = any(
                _esta_ativa(valor)
                for valor in atividade_linhas[inicio:inicio + len(bloco_df)]
            )
        blocos.append((bloco_df, np.mean(matriz, axis=0), atividade_bloco))

    acordes = list(acordes)
    n_linhas = len(df_compassos)
    n_blocos = len(blocos)

    # Aceita acordes_finais (um por linha) ou caminho (um por bloco).
    if len(acordes) == n_linhas:
        acordes_por_bloco = []
        pos_inicio = 0
        for bloco_df, _, _ in blocos:
            acordes_do_bloco = acordes[pos_inicio:pos_inicio + len(bloco_df)]
            if not acordes_do_bloco:
                raise ValueError("Não foi possível associar acordes aos blocos.")
            if any(a != acordes_do_bloco[0] for a in acordes_do_bloco):
                raise ValueError(
                    "Há acordes diferentes dentro de um bloco. Verifique se "
                    "tamanho_bloco corresponde ao usado na geração."
                )
            acordes_por_bloco.append(acordes_do_bloco[0])
            pos_inicio += len(bloco_df)
    elif len(acordes) == n_blocos:
        acordes_por_bloco = acordes
    else:
        raise ValueError(
            f"Foram recebidos {len(acordes)} acordes; eram esperados "
            f"{n_linhas} (um por linha) ou {n_blocos} (um por bloco)."
        )

    if progressao_original is not None:
        progressao_original = list(progressao_original)
        if len(progressao_original) != n_blocos:
            raise ValueError(
                f"progressao_original deve conter um acorde por bloco: "
                f"esperados {n_blocos}, recebidos {len(progressao_original)}."
            )

    linhas = []
    for numero_bloco, ((bloco_df, chroma, voz_ativa), acorde_escolhido) in enumerate(
        zip(blocos, acordes_por_bloco), start=1
    ):
        if acorde_escolhido not in campo:
            raise ValueError(
                f"O acorde '{acorde_escolhido}' não pertence ao campo harmônico "
                f"de {tom_raiz}. Verifique a tonalidade e os acordes."
            )

        energia_total = float(np.sum(chroma))
        if energia_total > 0:
            energia_acorde = sum(
                float(chroma[NOTE_NAMES.index(nota)])
                for nota in campo[acorde_escolhido]["notas"]
            )
            energia_cromatica = sum(
                float(energia)
                for indice, energia in enumerate(chroma)
                if indice not in indices_escala
            )
            energia_diatonica_fora = max(
                0.0, energia_total - energia_acorde - energia_cromatica
            )
            notas_acorde_pct = 100.0 * energia_acorde / energia_total
            notas_diatonicas_fora_pct = (
                100.0 * energia_diatonica_fora / energia_total
            )
            notas_cromaticas_pct = 100.0 * energia_cromatica / energia_total
        else:
            notas_acorde_pct = 0.0
            notas_diatonicas_fora_pct = 0.0
            notas_cromaticas_pct = 0.0

        custos = {
            nome: float(_custo_emissao(chroma, campo[nome], indices_escala))
            for nome in candidatos
        }
        melhores = sorted(custos.items(), key=lambda item: item[1])
        custo_escolhido = custos[acorde_escolhido]
        melhor_acorde, custo_melhor = melhores[0]
        diferenca_custo = custo_escolhido - custo_melhor

        motivos = []
        if voz_ativa is False:
            # Sem voz detectada, o chroma pode representar silêncio/ruído ou
            # valores de preenchimento; não emitir alertas de compatibilidade.
            motivo = "Sem atividade vocal; avaliação melódica não aplicada"
            revisar = False
        else:
            if notas_acorde_pct < limiar_notas_acorde:
                motivos.append("pouca energia nas notas do acorde")
            if notas_cromaticas_pct > limiar_notas_cromaticas:
                motivos.append("energia cromática elevada")
            if diferenca_custo > tolerancia_custo:
                motivos.append("outro acorde tem menor custo local")
            motivo = "; ".join(motivos) if motivos else "Sem alerta automático"
            revisar = bool(motivos)

        acorde_original = (
            progressao_original[numero_bloco - 1]
            if progressao_original is not None else None
        )
        corresponde_original = (
            str(acorde_escolhido).strip().casefold()
            == str(acorde_original).strip().casefold()
            if progressao_original is not None else None
        )

        linha = {
            "bloco": numero_bloco,
            "unidade_inicio": bloco_df.index[0],
            "unidade_fim": bloco_df.index[-1],
            "acorde": acorde_escolhido,
            "acorde_original": acorde_original,
            "corresponde_original": corresponde_original,
            "voz_ativa": voz_ativa,
            "custo_emissao": custo_escolhido,
            "melhor_acorde_local": melhor_acorde,
            "custo_melhor": custo_melhor,
            "diferenca_custo": diferenca_custo,
            "notas_acorde_pct": notas_acorde_pct,
            "notas_diatonicas_fora_pct": notas_diatonicas_fora_pct,
            "notas_cromaticas_pct": notas_cromaticas_pct,
            "melhores_candidatos": ", ".join(
                f"{nome} ({custo:.3f})" for nome, custo in melhores[:3]
            ),
            "revisar": revisar,
            "motivo": motivo,
        }

        for coluna in ("compasso", "unidade", "tempo_inicio", "tempo_fim"):
            if coluna in bloco_df.columns:
                linha[f"{coluna}_inicio"] = bloco_df[coluna].iloc[0]
                linha[f"{coluna}_fim"] = bloco_df[coluna].iloc[-1]

        linhas.append(linha)

    df_avaliacao = pd.DataFrame(linhas)
    # Estatísticas de compatibilidade consideram apenas blocos com voz ativa,
    # quando essa informação foi fornecida. Os custos por bloco continuam na tabela.
    if atividade_linhas is not None:
        avaliacao_resumo = df_avaliacao[df_avaliacao["voz_ativa"] == True]
    else:
        avaliacao_resumo = df_avaliacao

    resumo = {
        "unidades_avaliadas": int(len(df_avaliacao)),
        "unidades_com_voz_ativa": (
            int(df_avaliacao["voz_ativa"].eq(True).sum())
            if atividade_linhas is not None else None
        ),
        "custo_emissao_medio": round(float(avaliacao_resumo["custo_emissao"].mean()), 3)
            if not avaliacao_resumo.empty else None,
        "notas_do_acorde_pct_media": round(float(avaliacao_resumo["notas_acorde_pct"].mean()), 1)
            if not avaliacao_resumo.empty else None,
        "notas_diatonicas_fora_pct_media": round(
            float(avaliacao_resumo["notas_diatonicas_fora_pct"].mean()), 1
        ) if not avaliacao_resumo.empty else None,
        "notas_cromaticas_pct_media": round(
            float(avaliacao_resumo["notas_cromaticas_pct"].mean()), 1
        ) if not avaliacao_resumo.empty else None,
        "unidades_para_revisar": int(df_avaliacao["revisar"].sum()),
        "unidades_com_acorde_localmente_subotimo": int(
            (avaliacao_resumo["diferenca_custo"] > tolerancia_custo).sum()
        ),
    }
    if progressao_original is not None:
        resumo.update({
            "unidades_comparadas_com_original": int(n_blocos),
            "acordes_iguais_ao_original": int(df_avaliacao["corresponde_original"].sum()),
            "acordes_diferentes_do_original": int((~df_avaliacao["corresponde_original"]).sum()),
            "concordancia_com_original_pct": round(
                100.0 * float(df_avaliacao["corresponde_original"].mean()), 1
            ),
        })
    return df_avaliacao, resumo


# Compatibilidade com o nome usado em versões anteriores do projeto.
avaliar_compativibilidade = avaliar_compatibilidade

def analisar_chroma_bloco(
    df_compassos, num_bloco, tamanho_bloco=2, note_names=NOTE_NAMES
):
    """Exibe em detalhes a distribuição de energia média do vetor Chroma

    para um bloco agrupado de linhas do DataFrame (compassos inteiros ou
    unidades fracionárias, dependendo de como extract_chroma foi chamado).
    """
    if df_compassos.empty:
        print("O DataFrame fornecido está vazio.")
        return

    # Garante que os dados estejam ordenados (por unidade, se existir; senão por compasso)
    col_ordem = "unidade" if "unidade" in df_compassos.columns else "compasso"
    df_ordenado = df_compassos.sort_values(col_ordem).reset_index(drop=True)
    total_linhas = len(df_ordenado)

    # Calcula os índices inicial e final do bloco (base 1)
    inicio_idx = (num_bloco - 1) * tamanho_bloco
    fim_idx = inicio_idx + tamanho_bloco

    # Seleciona as linhas referentes ao bloco
    bloco = df_ordenado.iloc[inicio_idx:fim_idx]

    if bloco.empty:
        print(
            f"Bloco {num_bloco} não encontrado (limite de linhas excedido)."
        )
        return

    # Extrai os dados agregados do bloco
    itens_incluidos = bloco[col_ordem].tolist()
    tempo_inicio = bloco["tempo_inicio"].iloc[0]
    tempo_fim = bloco["tempo_fim"].iloc[-1]

    # Calcula a média dos vetores chroma no bloco
    chromas = np.array(bloco["chroma"].tolist())
    chroma_medio = np.mean(chromas, axis=0)

    # Identifica a nota dominante média do bloco
    nota_dominante_idx = np.argmax(chroma_medio)
    nota_dominante = note_names[nota_dominante_idx]

    print(
        f"\n=== ANÁLISE DETALHADA DO CHROMA - BLOCO {num_bloco} (Tamanho {tamanho_bloco}) ==="
    )
    print(f"{col_ordem.capitalize()}s Incluídos: {itens_incluidos}")
    print(f"Intervalo de Tempo: {tempo_inicio:.2f}s - {tempo_fim:.2f}s")
    print(
        f"Nota Dominante do Bloco: {nota_dominante} ({chroma_medio[nota_dominante_idx]*100:.2f}%)"
    )
    print("-" * 55)

    # Exibe a porcentagem de energia de cada uma das 12 notas no bloco
    for nota, energia in zip(note_names, chroma_medio):
        porcentagem = energia * 100
        barra = "█" * int(np.round(porcentagem / 5))
        print(f"{nota:>3}: {porcentagem:6.2f}% | {barra}")

    print("=" * 55)


def analisar_influencia_transicao(
    df_compassos,
    df_avaliacao,
    tom_raiz,
    caminho,
    tamanho_bloco=1,
    peso_transicao=1.2,
    permitir_diminutos=False,
    custo_repouso_final=0.0,
    custo_troca_fora_tempo_forte=0.0,
):
    """
    Analisa a influência das transições nas escolhas do Viterbi.

    Executa a análise somente nos blocos com revisar=True.

    Para cada bloco revisado:
      - Calcula o custo do caminho original.
      - Calcula o melhor caminho possível mantendo o acorde escolhido.
      - Calcula o melhor caminho possível para cada acorde alternativo.
      - Compara os custos de emissão, transição e repouso final.

    Não recalibra nem modifica os pesos do modelo.
    """

    import numpy as np
    import pandas as pd

    colunas_resultado = [
        "bloco",
        "acorde_escolhido",
        "melhor_acorde_local",
        "melhor_alternativa_global",
        "custo_global_escolhido",
        "custo_global_alternativa",
        "emissao_escolhido",
        "emissao_alternativa",
        "transicao_escolhido",
        "transicao_alternativa",
        "repouso_final_escolhido",
        "repouso_final_alternativa",
        "diferenca_emissao",
        "diferenca_transicao",
        "diferenca_repouso_final",
        "diferenca_global",
        "transicao_favoreceu_escolhido",
        "transicao_compensou_emissao",
        "caminho_escolhido",
        "caminho_alternativa",
    ]

    # ---------------------------------------------------------
    # 1. Verifica se existem blocos para revisar
    # ---------------------------------------------------------

    if "revisar" not in df_avaliacao.columns:
        raise ValueError(
            "df_avaliacao precisa possuir a coluna 'revisar'."
        )

    revisoes = df_avaliacao[
        df_avaliacao["revisar"].fillna(False).astype(bool)
    ]

    if revisoes.empty:
        print("Nenhum bloco marcado para revisão.")
        return pd.DataFrame(columns=colunas_resultado)

    if tamanho_bloco <= 0:
        raise ValueError("tamanho_bloco deve ser maior que zero.")

    if not caminho:
        raise ValueError("O caminho do Viterbi está vazio.")

    # ---------------------------------------------------------
    # 2. Prepara o campo harmônico
    # ---------------------------------------------------------

    campo = generate_harmonic_field(tom_raiz)

    _, indices_escala = get_scale_notes(tom_raiz)
    _, modo = extract_tone_and_mode(tom_raiz)

    funcao_map = (
        FUNCAO_MENOR if modo == "menor"
        else FUNCAO_MAIOR
    )

    funcoes = {
        nome: funcao_map.get(info["grau"], "T")
        for nome, info in campo.items()
    }

    nomes_acordes = list(campo.keys())

    if not permitir_diminutos:
        nomes_acordes = [
            nome for nome in nomes_acordes
            if "dim" not in nome
        ]

    # ---------------------------------------------------------
    # 3. Reconstrói os blocos e seus custos de emissão
    # ---------------------------------------------------------

    blocos = []

    for i in range(0, len(df_compassos), tamanho_bloco):

        bloco = df_compassos.iloc[i:i + tamanho_bloco]

        chroma_matrix = np.array(
            bloco["chroma"].tolist()
        )

        chroma_medio = np.mean(chroma_matrix, axis=0)

        posicao = (
            int(bloco["posicao_no_compasso"].iloc[0])
            if "posicao_no_compasso" in df_compassos.columns
            else 0
        )

        blocos.append((chroma_medio, posicao))

    n = len(blocos)

    if len(caminho) != n:
        raise ValueError(
            f"O caminho possui {len(caminho)} acordes, "
            f"mas foram encontrados {n} blocos."
        )

    # Matriz de emissão: bloco x acorde.
    emissao = []

    for chroma_medio, _ in blocos:

        custos = {}

        for nome in nomes_acordes:

            custos[nome] = _custo_emissao(
                chroma_medio,
                campo[nome],
                indices_escala
            )

        emissao.append(custos)

    # ---------------------------------------------------------
    # 4. Reproduz exatamente o custo de transição original
    # ---------------------------------------------------------

    def custo_transicao(a, b, posicao_destino):

        base = CUSTO_FUNCAO.get(
            (funcoes[a], funcoes[b]),
            0.5
        )

        if a == b:
            base += 0.3

        elif posicao_destino != 0:
            base += custo_troca_fora_tempo_forte

        return base * peso_transicao

    # ---------------------------------------------------------
    # 5. Custo de repouso final
    # ---------------------------------------------------------

    custo_terminal = {
        nome: (
            0.0 if funcoes[nome] == "T"
            else custo_repouso_final
        )
        for nome in nomes_acordes
    }

    # ---------------------------------------------------------
    # 6. Viterbi progressivo
    # ---------------------------------------------------------

    forward = [{} for _ in range(n)]
    anterior = [{} for _ in range(n)]

    for nome in nomes_acordes:

        forward[0][nome] = emissao[0][nome]
        anterior[0][nome] = None

    for t in range(1, n):

        posicao = blocos[t][1]

        for nome in nomes_acordes:

            melhor_custo = float("inf")
            melhor_prev = None

            for prev in nomes_acordes:

                total = (
                    forward[t - 1][prev]
                    + custo_transicao(prev, nome, posicao)
                    + emissao[t][nome]
                )

                if total < melhor_custo:
                    melhor_custo = total
                    melhor_prev = prev

            forward[t][nome] = melhor_custo
            anterior[t][nome] = melhor_prev

    # ---------------------------------------------------------
    # 7. Viterbi regressivo
    # ---------------------------------------------------------

    backward = [{} for _ in range(n)]
    proximo = [{} for _ in range(n)]

    for nome in nomes_acordes:

        backward[-1][nome] = custo_terminal[nome]
        proximo[-1][nome] = None

    for t in range(n - 2, -1, -1):

        posicao_destino = blocos[t + 1][1]

        for nome in nomes_acordes:

            melhor_custo = float("inf")
            melhor_next = None

            for nxt in nomes_acordes:

                total = (
                    custo_transicao(
                        nome,
                        nxt,
                        posicao_destino
                    )
                    + emissao[t + 1][nxt]
                    + backward[t + 1][nxt]
                )

                if total < melhor_custo:
                    melhor_custo = total
                    melhor_next = nxt

            backward[t][nome] = melhor_custo
            proximo[t][nome] = melhor_next

    # Custo global mínimo condicionado a cada acorde em cada bloco.
    custo_condicionado = [{} for _ in range(n)]

    for t in range(n):

        for nome in nomes_acordes:

            custo_condicionado[t][nome] = (
                forward[t][nome]
                + backward[t][nome]
            )

    # ---------------------------------------------------------
    # 8. Reconstrói o melhor caminho condicionado a um acorde
    # ---------------------------------------------------------

    def reconstruir_caminho(t, acorde):

        prefixo = [acorde]
        atual = acorde

        for k in range(t, 0, -1):

            atual = anterior[k][atual]
            prefixo.append(atual)

        prefixo.reverse()

        sufixo = []
        atual = acorde

        for k in range(t, n - 1):

            atual = proximo[k][atual]
            sufixo.append(atual)

        return prefixo + sufixo

    # ---------------------------------------------------------
    # 9. Decompõe os custos de uma sequência completa
    # ---------------------------------------------------------

    def decompor_caminho(seq):

        total_emissao = sum(
            emissao[t][nome]
            for t, nome in enumerate(seq)
        )

        total_transicao = sum(
            custo_transicao(
                seq[t - 1],
                seq[t],
                blocos[t][1]
            )
            for t in range(1, n)
        )

        repouso_final = custo_terminal[seq[-1]]

        total = (
            total_emissao
            + total_transicao
            + repouso_final
        )

        return {
            "emissao": total_emissao,
            "transicao": total_transicao,
            "repouso": repouso_final,
            "total": total,
        }

    # ---------------------------------------------------------
    # 10. Analisa somente os blocos marcados para revisão
    # ---------------------------------------------------------

    resultados = []

    for _, linha in revisoes.iterrows():

        idx = int(linha["bloco"]) - 1

        if idx < 0 or idx >= n:
            raise ValueError(
                f"Número de bloco inválido: {linha['bloco']}"
            )

        escolhido = linha["acorde"]
        melhor_local = linha["melhor_acorde_local"]

        if escolhido not in nomes_acordes:
            raise ValueError(
                f"Acorde escolhido inválido: {escolhido}"
            )

        if melhor_local not in nomes_acordes:
            raise ValueError(
                f"Melhor acorde local inválido: {melhor_local}"
            )

        if caminho[idx] != escolhido:
            raise ValueError(
                f"Inconsistência no bloco {idx + 1}: "
                f"avaliação indica {escolhido}, "
                f"mas o caminho contém {caminho[idx]}."
            )

        # Melhor caminho global que mantém o acorde escolhido.
        caminho_escolhido = reconstruir_caminho(
            idx, escolhido
        )

        custos_escolhido = decompor_caminho(
            caminho_escolhido
        )

        # Melhor alternativa global, excluindo o acorde escolhido.
        alternativas = [
            nome for nome in nomes_acordes
            if nome != escolhido
        ]

        melhor_alternativa = min(
            alternativas,
            key=lambda nome: custo_condicionado[idx][nome]
        )

        caminho_alternativa = reconstruir_caminho(
            idx, melhor_alternativa
        )

        custos_alternativa = decompor_caminho(
            caminho_alternativa
        )

        # Compara a alternativa com o acorde escolhido.
        # Valores positivos favorecem o acorde escolhido,
        # pois indicam que a alternativa custa mais.

        diferenca_emissao = (
            custos_alternativa["emissao"]
            - custos_escolhido["emissao"]
        )

        diferenca_transicao = (
            custos_alternativa["transicao"]
            - custos_escolhido["transicao"]
        )

        diferenca_repouso = (
            custos_alternativa["repouso"]
            - custos_escolhido["repouso"]
        )

        diferenca_global = (
            custos_alternativa["total"]
            - custos_escolhido["total"]
        )

        resultados.append({

            "bloco": idx + 1,

            "acorde_escolhido": escolhido,

            "melhor_acorde_local": melhor_local,

            "melhor_alternativa_global": melhor_alternativa,

            "custo_global_escolhido": (
                custo_condicionado[idx][escolhido]
            ),

            "custo_global_alternativa": (
                custo_condicionado[idx][melhor_alternativa]
            ),

            "emissao_escolhido": (
                custos_escolhido["emissao"]
            ),

            "emissao_alternativa": (
                custos_alternativa["emissao"]
            ),

            "transicao_escolhido": (
                custos_escolhido["transicao"]
            ),

            "transicao_alternativa": (
                custos_alternativa["transicao"]
            ),

            "repouso_final_escolhido": (
                custos_escolhido["repouso"]
            ),

            "repouso_final_alternativa": (
                custos_alternativa["repouso"]
            ),

            "diferenca_emissao": diferenca_emissao,

            "diferenca_transicao": diferenca_transicao,

            "diferenca_repouso_final": diferenca_repouso,

            "diferenca_global": diferenca_global,

            "transicao_favoreceu_escolhido": (
                diferenca_transicao > 0
            ),

            "transicao_compensou_emissao": (
                diferenca_emissao < 0
                and diferenca_transicao > 0
                and diferenca_global > 0
            ),

            "caminho_escolhido": caminho_escolhido,

            "caminho_alternativa": caminho_alternativa,

        })

    return pd.DataFrame(resultados, columns=colunas_resultado)
