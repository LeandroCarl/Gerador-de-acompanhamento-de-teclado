import numpy as np
import pandas as pd
from src.harmony.harmony import get_scale_notes, NOTE_NAMES

def extract_chroma(time, frequency, confidence, bpm, tom, beats_per_bar=4, conf_thresh=0, offset=0.0, quantizar_para_tom=False):
    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = seconds_per_beat * beats_per_bar
    t_anacruse = offset * seconds_per_beat

    # Processa o tom (suporta 'Am', 'F#m', 'C', 'Gmaj', etc.)
    tonica_idx, indices_escala = get_scale_notes(tom)

    total_duration = time[-1]
    total_bars = int(np.ceil((total_duration - t_anacruse) / seconds_per_bar))

    compassos = []

    for bar_idx in range(total_bars):
        t_start = t_anacruse + (bar_idx * seconds_per_bar)
        t_end = t_start + seconds_per_bar

        mask_bar = (time >= t_start) & (time < t_end)
        freqs_bar = frequency[mask_bar]
        confs_bar = confidence[mask_bar]

        # Filtra frequências com base na confiança e acima de 0 Hz
        valid_freqs = freqs_bar[(confs_bar >= conf_thresh) & (freqs_bar > 0)]
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
            "compasso": bar_idx + 1,
            "tempo_inicio": round(t_start, 2),
            "tempo_fim": round(t_end, 2),
            "chroma": chroma_vector,
            "nota_dominante": NOTE_NAMES[np.argmax(chroma_vector)]
        }
        compassos.append(item)

    return pd.DataFrame(compassos)

def analisar_chroma_bloco(
    df_compassos, num_bloco, tamanho_bloco=2, note_names=NOTE_NAMES
):
    """Exibe em detalhes a distribuição de energia média do vetor Chroma

    para um bloco agrupado de compassos (ex: tamanho_bloco=2).
    """
    if df_compassos.empty:
        print("O DataFrame fornecido está vazio.")
        return

    # Garante que os dados estejam ordenados pelo compasso
    df_ordenado = df_compassos.sort_values("compasso").reset_index(drop=True)
    total_compassos = len(df_ordenado)

    # Calcula os índices inicial e final do bloco (base 1)
    inicio_idx = (num_bloco - 1) * tamanho_bloco
    fim_idx = inicio_idx + tamanho_bloco

    # Seleciona as linhas referentes ao bloco
    bloco = df_ordenado.iloc[inicio_idx:fim_idx]

    if bloco.empty:
        print(
            f"Bloco {num_bloco} não encontrado (limite de compassos excedido)."
        )
        return

    # Extrai os dados agregados do bloco
    compassos_incluidos = bloco["compasso"].tolist()
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
    print(f"Compassos Incluídos: {compassos_incluidos}")
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