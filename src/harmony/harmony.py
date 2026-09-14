import numpy as np

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

ESCALAS = {
    'maior': [0, 2, 4, 5, 7, 9, 11],
    'menor': [0, 2, 3, 5, 7, 8, 10]
}
# Estrutura dos graus do Campo Harmônico Maior (Sufixo e Intervalos de Tônica, Terça e Quinta)
MAJOR_KEY_TRIADS = [
    {"grau": "I",   "sufixo": "",   "intervalos": [0, 4, 7]},   # Maior (ex: C)
    {"grau": "ii",  "sufixo": "m",  "intervalos": [2, 5, 9]},   # Menor (ex: Dm)
    {"grau": "iii", "sufixo": "m",  "intervalos": [4, 7, 11]},  # Menor (ex: Em)
    {"grau": "IV",  "sufixo": "",   "intervalos": [5, 9, 0]},   # Maior (ex: F)
    {"grau": "V",   "sufixo": "",   "intervalos": [7, 11, 2]},  # Maior (ex: G)
    {"grau": "vi",  "sufixo": "m",  "intervalos": [9, 0, 4]},   # Menor (ex: Am)
    {"grau": "vii°","sufixo": "dim","intervalos": [11, 2, 5]}   # Diminuto (ex: Bdim)
]

MINOR_KEY_TRIADS = [
    {"grau": "i",   "sufixo": "m",  "intervalos": [0, 3, 7]},   # Menor (Tônica)
    {"grau": "ii°", "sufixo": "dim","intervalos": [2, 5, 8]},   # Diminuto
    {"grau": "III", "sufixo": "",   "intervalos": [3, 7, 10]},  # Maior (Relativo)
    {"grau": "iv",  "sufixo": "m",  "intervalos": [5, 8, 0]},   # Menor
    {"grau": "V",   "sufixo": "",   "intervalos": [7, 11, 2]},  # Maior (Dominante Harmônica)
    {"grau": "VI",  "sufixo": "",   "intervalos": [8, 0, 3]},   # Maior
    {"grau": "VII", "sufixo": "",   "intervalos": [10, 2, 5]}   # Maior
]

# ---------------------------------------------------------------
# 1. Mapa de função tonal por grau (reaproveita o campo 'grau' que
#    já vem de MAJOR_KEY_TRIADS / MINOR_KEY_TRIADS)
# ---------------------------------------------------------------
FUNCAO_MAIOR = {
    "I": "T", "ii": "SD", "iii": "T", "IV": "SD",
    "V": "D", "vi": "T", "vii°": "D",
}
FUNCAO_MENOR = {
    "i": "T", "ii°": "SD", "III": "T", "iv": "SD",
    "V": "D", "VI": "T", "VII": "SD",
}

def get_scale_notes(tom_str):
    """Extrai a tônica e retorna os 7 índices cromáticos pertencentes à escala."""
    tom_str = tom_str.strip()
    is_menor = tom_str.endswith('m') or 'min' in tom_str.lower()
    
    # Remove sufixos para isolar a nota base
    tonica_nome = tom_str.replace('min', '').replace('m', '').replace('M', '').upper()
    tonica_idx = NOTE_NAMES.index(tonica_nome)
    
    tipo_escala = 'menor' if is_menor else 'maior'
    intervalos = ESCALAS[tipo_escala]
    
    # Calcula os índices das 7 notas da escala dentro do vetor de 12 posições
    indices_escala = [(tonica_idx + i) % 12 for i in intervalos]
    return tonica_idx, set(indices_escala)

def extract_tone_and_mode(tom_str):
    """
    Recebe strings como 'Em', 'C#m', 'Bb', 'G', 'Am'
    e retorna uma tupla: (raiz, modo)
    Ex: 'Em' -> ('E', 'menor')
        'C#m' -> ('C#', 'menor')
        'G' -> ('G', 'maior')
    """
    tom_clean = tom_str.strip()

    # Verifica se é menor (termina com 'm' ou 'min', mas ignora 'dim')
    if (tom_clean.endswith('m') or tom_clean.endswith('min')) and not tom_clean.endswith('dim'):
        modo = "menor"
        # Remove o 'm' ou 'min' do final para isolar a raiz
        raiz = tom_clean.rstrip('min').rstrip('m')
    else:
        modo = "maior"
        raiz = tom_clean

    # Normaliza a primeira letra em maiúscula
    raiz = raiz.capitalize()

    return raiz, modo

def generate_harmonic_field(tom_entrada):
    """
    Gera os 7 acordes do campo harmônico (Maior ou Menor).
    Exemplo de tonalidade_raiz: 'C', 'G', 'A'
    Exemplo de modo: 'maior' ou 'menor'
    """
    tonalidade_raiz, modo = extract_tone_and_mode(tom_entrada)
    root_idx = NOTE_NAMES.index(tonalidade_raiz.upper())
    campo = {}

    # Escolhe a tabela de intervalos com base no modo informado
    triadas = MINOR_KEY_TRIADS if modo.lower() in ["menor", "minor", "m"] else MAJOR_KEY_TRIADS

    for triada in triadas:
        root_acorde = NOTE_NAMES[(root_idx + triada["intervalos"][0]) % 12]
        nome_acorde = f"{root_acorde}{triada['sufixo']}"

        notas_do_acorde = [
            NOTE_NAMES[(root_idx + interval) % 12] for interval in triada["intervalos"]
        ]

        campo[nome_acorde] = {
            "grau": triada["grau"],
            "notas": notas_do_acorde
        }
    return campo


# Custo de ir de uma função para outra (assimétrico, de propósito:
# harmonia tonal tem "sentido" — dominante puxa para tônica, mas
# tônica não puxa para dominante com a mesma força).
CUSTO_FUNCAO = {
    ("T", "T"): 0.4, ("T", "SD"): 0.2, ("T", "D"): 0.5,
    ("SD", "T"): 0.3, ("SD", "SD"): 0.6, ("SD", "D"): 0.1,
    ("D", "T"): 0.0, ("D", "SD"): 1.6, ("D", "D"): 0.7,
}