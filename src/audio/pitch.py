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

def listen_beats(y, sr, bpm, beats_per_bar=4, anacruse_beats=0.0):
    duracao_total = len(y) / sr
    tempo_batida = 60.0 / bpm
    
    # 1. Calcula quando o PRIMEIRO tempo forte (Downbeat) acontece no áudio
    t_primeiro_downbeat = anacruse_beats * tempo_batida

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

def cut_vocal(
    caminho_audio, 
    bpm=90, 
    compasso_inicio=1, 
    compasso_fim=4, 
    beats_per_bar=4,
    offset=0.0,  # 1 tempo de anacruse (duas colcheias no tempo 4)
    sr=22050
):
    """
    Recorta o áudio vocal dos compassos informados, no bpm informado, 
    garantindo que a grade considere um deslocamento inicial.
    """
    y, sr = librosa.load(caminho_audio, sr=sr, mono=True)
    
    segundo_por_tempo = 60.0 / bpm
    duracao_compasso = segundo_por_tempo * beats_per_bar

    # O primeiro tempo forte (downbeat) ocorre APÓS o tempo da anacruse
    t_primeiro_downbeat = offset * segundo_por_tempo

    # O recorte para o Compasso 1 começa no tempo forte (ou inclui a anacruse, dependendo do objetivo)
    tempo_inicio_sec = (compasso_inicio - 1) * duracao_compasso + t_primeiro_downbeat
    tempo_fim_sec = compasso_fim * duracao_compasso + t_primeiro_downbeat

    # Converte em amostras (samples)
    s_inicio = int(tempo_inicio_sec * sr)
    s_fim = int(tempo_fim_sec * sr)

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