import numpy as np
import pandas as pd
from src.harmony.harmony import get_scale_notes, NOTE_NAMES


def extract_chroma(
    time,
    frequency,
    confidence,
    bpm,
    tom,
    beats_per_bar=4,
    unidades_por_compasso=1,
    conf_thresh=0,
    offset=0.0,
    quantizar_para_tom=False,
):
    """Extrai o vetor Chroma por janela de tempo.
    Cada linha do DataFrame retornado corresponde a uma unidade e inclui:
        - "unidade": índice global da unidade (1-based)
        - "compasso": número do compasso ao qual a unidade pertence (1-based)
        - "posicao_no_compasso": posição da unidade dentro do compasso
          (0 = início do compasso / tempo forte; >0 = subdivisão seguinte)
        - "chroma": Vetor de 12 posiçoes contendo as proporçoes de cada nota
        em uma dada amostra do áudio 
        - "nota_dominante": nota que predominou em uma amostra coletada
    """
    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = seconds_per_beat * beats_per_bar
    seconds_per_unidade = seconds_per_bar / unidades_por_compasso
    t_anacruse = offset * seconds_per_beat

    # Processa o tom (suporta 'Am', 'F#m', 'C', 'Gmaj', etc.)
    tonica_idx, indices_escala = get_scale_notes(tom)

    total_duration = time[-1]
    total_unidades = int(np.ceil((total_duration - t_anacruse) / seconds_per_unidade))

    unidades = []

    for uidx in range(total_unidades):
        t_start = t_anacruse + (uidx * seconds_per_unidade)
        t_end = t_start + seconds_per_unidade

        mask_unidade = (time >= t_start) & (time < t_end)
        freqs_u = frequency[mask_unidade]
        confs_u = confidence[mask_unidade]

        # Filtra frequências com base na confiança e acima de 0 Hz
        valid_freqs = freqs_u[(confs_u >= conf_thresh) & (freqs_u > 0)]
        chroma_vector = np.zeros(12)

        if len(valid_freqs) > 0:
            raw_midis = 69 + 12 * np.log2(valid_freqs / 440.0)
            for m in raw_midis:
                pitch_class = int(np.round(m)) % 12

                # Se quantizar_para_tom=True, ignora notas fora da escala
                if quantizar_para_tom:
                    if pitch_class in indices_escala:
                        chroma_vector[pitch_class] += 1
                else:
                    chroma_vector[pitch_class] += 1

            soma = chroma_vector.sum()
            if soma > 0:
                chroma_vector /= soma
            else:
                # Se todas as notas detectadas estavam fora da escala, usa a tônica
                chroma_vector[tonica_idx] = 1.0
        else:
            # Fallback para silêncios
            chroma_vector[tonica_idx] = 1.0

        item = {
            "unidade": uidx + 1,
            "compasso": (uidx // unidades_por_compasso) + 1,
            "posicao_no_compasso": uidx % unidades_por_compasso,
            "tempo_inicio": round(t_start, 2),
            "tempo_fim": round(t_end, 2),
            "chroma": chroma_vector,
            "nota_dominante": NOTE_NAMES[np.argmax(chroma_vector)],
        }
        unidades.append(item)

    return pd.DataFrame(unidades)


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