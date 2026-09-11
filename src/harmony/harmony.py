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
