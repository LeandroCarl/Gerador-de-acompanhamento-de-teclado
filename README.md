# Gerador-de-acompanhamento-de-teclado
Um sistema que gera acompanhamento de teclado automático que rastreia a voz cantada e usa o viterbi para sugerir a progressão de acordes que mais se encaixa com a melodia de entrada, levando em conta regras da gramática tonal. 

## Funcionalidades
- Extração de Chroma Vocal: Análise de sinal de áudio vocal para extração do vetor de energia das 12 notas musicais por compasso.
- Modelo de Emissão Hierárquico em 3 Níveis: Diferenciação de pesos entre notas do acorde, notas de passagem da melodia vocal e ruídos/notas fora do tom.
- Decodificador Harmônico Viterbi: Escolha global e otimizada da sequência de acordes baseada em menor custo acumulado.
- Gramática Funcional Tonal: Matriz de custos de transição baseada em funções harmônicas (Tônica - T, Subdominante - SD, Dominante - D).
- Geração de Arranjo e Condução de Vozes (Voice Leading): Tradução dos acordes em notas reais de teclado com inversões suaves e acompanhamento rítmico.
- Renderização de Áudio e MIDI: Exportação dos resultados em arquivos .mid ou sintetizados em áudio (.wav) via SoundFonts.
## Modelo de Custos
Foi utilizado um modelo de custos em cada nota cantada possui um peso, que varia conforme sua compatibilidade com a tonalidade informada e posição no acorde (se é uma tônica, terça ou quinta). Além disso, as transições de entre dois acordes também recebe um peso, cujo valor respeita regras da gramática funcional. 

### 1. Custos de Emissão 
A emissão mede o quão bem a energia do vetor Chroma se encaixa na estrutura de um acorde em determinado compasso. A penalidade por energia obedece a três níveis:
| Nível | Categoria de Nota | Peso | Descrição |
| :---: | :---: | ---: | :--- |
| **1** | **Tríade do Acorde** | **0.5** | Tônica (0.5), Terça (1.0) e Quinta (1.5) |
| **2** | **Nota Diatônica** | **3.5** | Notas pertencentes ao tom raiz |
| **3** | **Nota Cromática** | **12.0** | Frequências fora da escala da música |
Nota: Caso o próprio acorde testado não pertença ao campo harmônico da escala raiz, adiciona-se uma penalidade fixa de +1.5 no custo do bloco.

### 2. Custos de transição
A transição avalia a fluidez harmônica entre os blocos $t-1$ e $t$. A matriz base de custos opera sobre a função tonal do acorde (T, SD, D):
- Cadência Autêntica ($D \rightarrow T$): Custo 0.0 (Resolução ideal).
- Manutenção da Função ($T \rightarrow T$, $SD \rightarrow SD$, $D \rightarrow D$): Custo 0.0 a 0.1.
- Movimentos Padrão ($T \rightarrow SD$, $SD \rightarrow D$): Custo 0.2 a 0.3.
- Retrocesso Harmônico ($D \rightarrow SD$): Custo 2.0 (Penalizado por violar a condução funcional tradicional).

## Estrutura do Repositório
```text
gerador-de-acompanhamento-de-teclado/
├── data/                  # Áudios de teste, arquivos .wav e SoundFonts (.sf2)
├── notebooks/             # Experimentos em Jupyter Notebooks e visualizações de Chroma
├── src/
│   ├── audio/             # Processamento do sinal e extração de Chroma/Pitch
│   ├── harmony/           # Campo harmônico, funções tonais e algoritmo Viterbi
│   ├── arranger/          # Voicings de teclado, condução de vozes e ritmos
│   └── synth/             # Geração de MIDI e renderização com SoundFont
├── tests/                 # Testes unitários das funções e matrizes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```
## Pré-requisitos e Instalação
### Pré-requisitos
- Python 3.9+
### Instalação 

1. Clone o repositório:
   ```Bash
   git clone https://github.com/seu-usuario/Gerador-de-acompanhamento-de-teclado.git
   cd Gerador-de-acompanhamento-de-teclado
   ```
2. Crie e ative um ambiente python:
   ```bash
   python -m venv .venv
    # Linux/macOS:
    source .venv/bin/activate
    # Windows:
    .venv\Scripts\activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Como Usar
**Exemplo rápido**:
