import librosa
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt

def extract_f0_pyin(y, sr, fmin=65.0, fmax=1046.0, frame_length=2048, hop_length=512):
    """
    Extrai a frequência fundamental (F0), status de unvoiced e confiança da nota
    de um sinal vocal utilizando o algoritmo pYIN do Librosa.
    
    Retorna:
    - time: vetor de tempo para cada frame em segundos.
    - frequency: frequências F0 estimadas (Hz), com NaNs onde não há voz.
    - confidence: grau de certeza do pYIN na detecção do pitch (0.0 a 1.0).
    """
    # 1. Executa o pYIN no sinal de áudio
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # 2. Gera os timestamps em segundos para cada frame extraído
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    
    # 3. Trata valores nulos (NaN) para não quebrar cálculos posteriores
    # Substitui NaNs na frequência por 0.0 Hz
    frequency_clean = np.nan_to_num(f0, nan=0.0)
    
    # Substitui NaNs nas probabilidades por 0.0
    confidence_clean = np.nan_to_num(voiced_probs, nan=0.0)

    return times, frequency_clean, confidence_clean

def listen_beats(y, sr, bpm, beats_per_bar=4, offset=0.0):
    duracao_total = len(y) / sr
    tempo_batida = 60.0 / bpm
    
    # 1. Calcula quando o PRIMEIRO tempo forte (Downbeat) acontece no áudio
    t_primeiro_downbeat = offset * tempo_batida

    # 2. Gera os tempos fortes (1500 Hz) a partir do primeiro downbeat
    beat_times_fortes = np.arange(t_primeiro_downbeat, duracao_total, tempo_batida * beats_per_bar)

    # 3. Gera TODOS os tempos da música (fortes + fracos)
    # Se houver anacruse, gera batidas fracos também no trecho antes do tempo 1
    inicio_da_grade = t_primeiro_downbeat % tempo_batida
    beat_times_todos = np.arange(inicio_da_grade, duracao_total, tempo_batida)

    # 4. Os tempos fracos (800 Hz) são todos os tempos EXCETO os tempos fortes
    # Usamos tolerancia de ponto flutuante para comparar os timestamps
    beat_times_fracos = np.array([
        t for t in beat_times_todos 
        if not np.any(np.isclose(t, beat_times_fortes, atol=1e-4))
    ])

    # 5. Gera o sinal de áudio dos clicks agudos (Fortes)
    clicks_fortes = np.zeros(len(y))
    if len(beat_times_fortes) > 0:
        clicks_fortes = librosa.clicks(
            times=beat_times_fortes, 
            sr=sr, 
            click_freq=1500.0, 
            length=len(y)
        )

    # 6. Gera o sinal de áudio dos clicks graves (Fracos)
    clicks_fracos = np.zeros(len(y))
    if len(beat_times_fracos) > 0:
        clicks_fracos = librosa.clicks(
            times=beat_times_fracos, 
            sr=sr, 
            click_freq=800.0, 
            length=len(y)
        )

    # Combina ambos os clicks e exibe o áudio
    clicks_totais = clicks_fortes + (clicks_fracos * 0.7)
    audio_mix = y + clicks_totais * 0.5
    
    ipd.display(ipd.Audio(audio_mix, rate=sr))

def extract_vocal_activity(
    times,
    frequency,
    confidence,
    bpm,
    offset=0.0,
    beats_per_bar=4,
    conf_thresh=0.0,
    min_voiced_ratio=0.15,
    total_duration=None,
):
    """Calcula a atividade vocal por compasso utilizando as métricas extraídas pelo pYIN.

    Args:
        times (np.ndarray): Vetor de timestamps em segundos dos frames do pYIN.
        frequency (np.ndarray): Vetor de frequências F0 (Hz).
        confidence (np.ndarray): Probabilidade/grau de confiança do tom.
        bpm (float): Tempos por minuto.
        beats_per_bar (int): Batidas por compasso.
        conf_thresh (float): Limiar mínimo de confiança para considerar o frame
          como voz válida.
        min_voiced_ratio (float): Proporção mínima de frames com voz no
          compasso (ex: 0.15 = 15%) para considerar o compasso ativo.
        total_duration (float, optional): Duração total em segundos. Se None,
          usa o último timestamp.

    Returns:
        list[bool]: Lista booleana com True para compassos com canto e False
        para pausas.
    """
    duracao_compasso_seg = (60.0 / bpm) * beats_per_bar
    t_anacruse = offset * (60.0 / bpm)
    # Determina a duração total para calcular a quantidade de compassos
    if total_duration is None:
        total_duration = times[-1] if len(times) > 0 else 0.0

    total_compassos = int(np.ceil((total_duration - t_anacruse)/ duracao_compasso_seg))
    atividade = []

    for i in range(total_compassos):
        t_inicio = t_anacruse + i * duracao_compasso_seg
        t_fim = t_inicio + duracao_compasso_seg

        # Recorta os frames do pYIN pertencentes ao compasso atual
        mask_compasso = (times >= t_inicio) & (times < t_fim)
        freq_compasso = frequency[mask_compasso]
        conf_compasso = confidence[mask_compasso]

        if len(freq_compasso) == 0:
            atividade.append(False)
            continue

        # Identifica frames onde há frequência válida e confiança acima do limiar
        frames_com_voz = (freq_compasso > 0.0) & (conf_compasso >= conf_thresh)

        # Ratio de frames cantados em relação ao total de frames do compasso
        razao_cantada = np.sum(frames_com_voz) / len(freq_compasso)

        # O compasso é considerado ativo se a proporção ultrapassar o mínimo configurado
        atividade.append(razao_cantada >= min_voiced_ratio)

    return atividade

def cut_vocal(
    caminho_audio,
    bpm=90,
    compasso_inicio=1,
    compasso_fim=4,
    beats_per_bar=4,
    offset=0.0,
    detect_first_onset=True,
    sr=22050,
):
    """Recorta o áudio vocal alinhando a grade temporal ao primeiro ataque vocal detectado.

    Args:
        caminho_audio (str): Caminho do arquivo de áudio vocal.
        bpm (float): Tempos por minuto.
        compasso_inicio (int): Compasso inicial do recorte (base 1).
        compasso_fim (int): Compasso final do recorte.
        beats_per_bar (int): Batidas por compasso.
        offset (float): Deslocamento manual em tempos (ex: anacruse).
        detect_first_onset (bool): Se True, localiza o início real do som vocal
          via Librosa.
        sr (int): Taxa de amostragem.

    Returns:
        tuple[np.ndarray, int]: Áudio recortado e taxa de amostragem.
    """
    y, sr = librosa.load(caminho_audio, sr=sr, mono=True)

    segundo_por_tempo = 60.0 / bpm
    duracao_compasso = segundo_por_tempo * beats_per_bar

    # Ponto de partida padrão baseado apenas no offset teórico
    t_referencia = offset * segundo_por_tempo

    if detect_first_onset:
        # 1. Calcula a curva de força de ataques (onset envelope)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # 2. Localiza os tempos de onset. backtrack=True busca o vale de energia
        # imediatamente anterior ao pico, ideal para vozes que sobram no ataque.
        onsets = librosa.onset.onset_detect(
            y=y, sr=sr, onset_envelope=onset_env, units="time", backtrack=True
        )

        # 3. Alinha o primeiro tempo forte ao instante do primeiro ataque detectado
        if len(onsets) > 0:
            primeiro_onset_sec = onsets[0]
            t_referencia = primeiro_onset_sec + (offset * segundo_por_tempo)

    # Cálculo dos limites de tempo do corte
    tempo_inicio_sec = (
        compasso_inicio - 1
    ) * duracao_compasso + t_referencia
    tempo_fim_sec = compasso_fim * duracao_compasso + t_referencia

    # Converte para índices do vetor e aplica travas de segurança
    s_inicio = max(0, int(tempo_inicio_sec * sr))
    s_fim = min(len(y), int(tempo_fim_sec * sr))

    y_recortado = y[s_inicio:s_fim]

    return y_recortado, sr

def show_spectrum(time, frequency, confident, conf_thresh, y, sr):
  filtered_frequency = np.where((confident > conf_thresh) & (frequency > 0), frequency, np.nan)
  plt.figure(figsize=(14, 6))
  D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
  librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='magma')
  plt.plot(time, filtered_frequency, color='cyan', linewidth=3.0, label='F0 do PYIN')
  plt.ylim(80, 1000)
  plt.title('Validação visual')
  plt.legend(loc='upper right')
  plt.colorbar(format='%.2f')
  plt.show()