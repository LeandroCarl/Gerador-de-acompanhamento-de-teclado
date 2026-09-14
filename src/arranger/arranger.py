from src.arranger.rhythm import generate_quarter_note_block
from src.arranger.voicing import (
    generate_midi_inversions,
    select_best_voicing,
)
from src.harmony.harmony import generate_harmonic_field, NOTE_NAMES


def arrange_chord_sequence(
    acordes_sugeridos, tom_raiz, bpm=120, beats_per_bar=4
):
    """Recebe a lista de acordes gerada pelo Viterbi e converte em eventos MIDI completos com condução de vozes e ritmo.

    Args:
        acordes_sugeridos (list[str]): Lista de nomes dos acordes (ex: ['C',
          'Am', 'Dm', 'G']).
        tom_raiz (str): Tom da música (ex: 'C').
        bpm (int): Tempos por minuto.

    Returns:
        list[dict]: Todos os eventos MIDI ordenados por tempo de início.
    """
    campo_harmonico = generate_harmonic_field(tom_raiz)
    todos_eventos = []
    voicing_anterior = None

    for bar_idx, nome_acorde in enumerate(acordes_sugeridos):
        info_acorde = campo_harmonico[nome_acorde]
        # 1. Mão Esquerda: Tônica na oitava 3 (MIDI 48-59) ou oitava 2 (MIDI 36-47)
        notas_indices = [NOTE_NAMES.index(n) for n in info_acorde['notas']]
        nota_baixo = notas_indices[0] + 36
        # 2. Mão Direita: Calcula a melhor inversão no registro médio
        inversoes = generate_midi_inversions(notas_indices)
        melhor_voicing = select_best_voicing(voicing_anterior, inversoes)
        voicing_anterior = melhor_voicing

        # 3. Ritmo: Aplica o padrão de semínimas para o compasso
        eventos_compasso = generate_quarter_note_block(
            bar_idx=bar_idx,
            nota_baixo=nota_baixo,
            voicing_mao_direita=melhor_voicing,
            bpm=bpm,
            beats_per_bar=beats_per_bar,
        )

        todos_eventos.extend(eventos_compasso)

    return todos_eventos