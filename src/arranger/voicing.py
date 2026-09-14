import numpy as np

def generate_midi_inversions(notas_acorde_idx):
    """Gera a posição fundamental e 2 inversões da mão direita na oitava 4 (MIDI 60-71)."""
    base = np.array(sorted(list(notas_acorde_idx))) + 60
    
    inv0 = np.sort(base)
    inv1 = np.sort([base[1], base[2], base[0] + 12])
    inv2 = np.sort([base[2], base[0] + 12, base[1] + 12])
    
    return [inv0, inv1, inv2]

def select_best_voicing(voicing_anterior, inversoes_disponiveis):
    """Seleciona a inversão que minimiza o movimento das notas da mão direita."""
    if voicing_anterior is None:
        return inversoes_disponiveis[0]

    melhor_inv = None
    menor_distancia = float('inf')

    for inv in inversoes_disponiveis:
        distancia = np.sum(np.abs(inv - voicing_anterior))
        if distancia < menor_distancia:
            menor_distancia = distancia
            melhor_inv = inv

    return melhor_inv