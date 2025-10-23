#!/usr/bin/env python3
"""
Visualizador Avançado de Sinais e Sistemas II - Versão 3.0
Com análises matemáticas completas e reprodução de áudio
Autor: Sistema Educacional de Processamento de Sinais
"""

import customtkinter as ctk
from tkinter import filedialog
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import signal, fft
from scipy.io import wavfile
from scipy.linalg import svd
import warnings
from typing import Tuple
from enum import Enum
import pyaudio
import threading
import queue
import time
from collections import deque

warnings.filterwarnings("ignore")

# Configuração do tema
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ==================== Classes de Dados ====================


class SignalType(Enum):
    """Tipos de sinais com descrições"""

    SINE = ("Senoidal", "x(t) = A·sin(2πft + φ)", 440, 0.5)
    SQUARE = ("Quadrada", "Série de Fourier: Σ(4/nπ)sin(nωt), n ímpar", 200, 0.05)
    SAWTOOTH = ("Dente de Serra", "Série de Fourier: Σ(2/nπ)sin(nωt)", 150, 0.05)
    TRIANGLE = ("Triangular", "Série de Fourier: Σ(8/n²π²)sin(nωt), n ímpar", 200, 0.05)
    CHIRP = ("Chirp", "f(t) = f0 + (f1-f0)·t/T", 100, 1.0)
    IMPULSE = ("Impulso δ[n]", "δ[n] = 1 se n=0, 0 caso contrário", 0, 0.1)
    STEP = ("Degrau u[n]", "u[n] = 1 se n≥0, 0 caso contrário", 0, 0.1)
    NOISE = ("Ruído Branco", "E[X]=0, Var[X]=σ², PSD constante", 0, 0.5)
    CUSTOM = ("Áudio", "Arquivo de áudio carregado", 0, 1.0)


# ==================== Processador de Sinais ====================


class AudioPlayer:
    """Reprodutor de áudio em tempo real"""

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.playing = False
        self.audio_queue = queue.Queue()
        self.play_thread = None

    def play_signal(self, signal_data: np.ndarray, fs: int):
        """Reproduz um sinal de áudio"""
        self.stop()

        # Normalizar e converter para int16
        normalized = signal_data / (np.max(np.abs(signal_data)) + 1e-10)
        audio_data = (normalized * 32767).astype(np.int16)

        # Criar stream
        self.stream = self.p.open(
            format=pyaudio.paInt16, channels=1, rate=fs, output=True
        )

        self.playing = True
        self.play_thread = threading.Thread(
            target=self._play_worker, args=(audio_data, fs)
        )
        self.play_thread.start()

    def _play_worker(self, audio_data, fs):
        """Worker thread para reprodução"""
        chunk_size = 1024
        for i in range(0, len(audio_data), chunk_size):
            if not self.playing:
                break
            chunk = audio_data[i : i + chunk_size]
            self.stream.write(chunk.tobytes())
        # Encerrar sem chamar stop() para não fazer join no mesmo thread
        self.playing = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            finally:
                self.stream = None
        self.play_thread = None

    def stop(self):
        """Para a reprodução"""
        self.playing = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        # Evita RuntimeError de join no próprio thread
        if (
            getattr(self, "play_thread", None)
            and threading.current_thread() is not self.play_thread
        ):
            self.play_thread.join(timeout=0.5)
        self.play_thread = None

    def __del__(self):
        """Cleanup"""
        self.stop()
        self.p.terminate()


class SignalProcessor:
    """Processamento avançado de sinais digitais"""

    def __init__(self):
        self.signal_data = None
        self.filter_coeffs = None

    def generate_signal(
        self,
        signal_type: SignalType,
        fs: int,
        duration: float,
        freq: float,
        amplitude: float,
        phase: float = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gera sinais com parâmetros especificados"""
        t = np.linspace(0, duration, int(fs * duration), False)

        if signal_type == SignalType.SINE:
            y = amplitude * np.sin(2 * np.pi * freq * t + phase)

        elif signal_type == SignalType.SQUARE:
            y = amplitude * signal.square(2 * np.pi * freq * t)

        elif signal_type == SignalType.SAWTOOTH:
            y = amplitude * signal.sawtooth(2 * np.pi * freq * t)

        elif signal_type == SignalType.TRIANGLE:
            y = amplitude * signal.sawtooth(2 * np.pi * freq * t, width=0.5)

        elif signal_type == SignalType.CHIRP:
            f1 = min(fs / 2, freq * 10)  # Frequência final
            y = amplitude * signal.chirp(t, freq, duration, f1, method="linear")

        elif signal_type == SignalType.IMPULSE:
            y = np.zeros_like(t)
            y[len(t) // 2] = amplitude

        elif signal_type == SignalType.STEP:
            y = np.zeros_like(t)
            y[len(t) // 2 :] = amplitude

        elif signal_type == SignalType.NOISE:
            y = amplitude * np.random.randn(len(t))

        else:
            y = np.zeros_like(t)

        return t, y

    def compute_fft(
        self, signal_data: np.ndarray, fs: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula FFT retornando frequências, magnitude e fase"""
        if len(signal_data) == 0:
            return np.array([]), np.array([]), np.array([])

        n = len(signal_data)
        fft_vals = fft.fft(signal_data)
        fft_freq = fft.fftfreq(n, 1 / fs)

        # Apenas frequências positivas
        idx = fft_freq >= 0
        freq = fft_freq[idx]
        magnitude = np.abs(fft_vals[idx]) * 2 / n  # Normalizar
        phase = np.angle(fft_vals[idx])

        return freq, magnitude, phase

    def compute_spectrogram(
        self, signal_data: np.ndarray, fs: int, window_size: int = 256
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula espectrograma com janelamento adequado"""
        # Ajustar tamanho da janela
        nperseg = min(window_size, len(signal_data) // 8)
        nperseg = max(16, nperseg)

        # Overlap de 50%
        noverlap = nperseg // 2

        # Calcular espectrograma
        f, t, Sxx = signal.spectrogram(
            signal_data,
            fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
        )

        # Converter para dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        return f, t, Sxx_db

    def design_filter(
        self, filter_type: str, fs: int, order: int, cutoff: float, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Projeta filtros digitais IIR"""
        nyquist = fs / 2

        try:
            if filter_type == "Passa-Baixa":
                wn = cutoff / nyquist
                b, a = signal.butter(order, wn, btype="low")

            elif filter_type == "Passa-Alta":
                wn = cutoff / nyquist
                b, a = signal.butter(order, wn, btype="high")

            elif filter_type == "Passa-Banda":
                low = kwargs.get("low", cutoff - 200)
                high = kwargs.get("high", cutoff + 200)
                wn = [low / nyquist, high / nyquist]
                wn = np.clip(wn, 0.01, 0.99)
                b, a = signal.butter(order, wn, btype="band")

            elif filter_type == "Rejeita-Banda":
                low = kwargs.get("low", cutoff - 100)
                high = kwargs.get("high", cutoff + 100)
                wn = [low / nyquist, high / nyquist]
                wn = np.clip(wn, 0.01, 0.99)
                b, a = signal.butter(order, wn, btype="stop")

            else:
                b, a = np.array([1.0]), np.array([1.0])

        except Exception as e:
            print(f"Erro no projeto do filtro: {e}")
            b, a = np.array([1.0]), np.array([1.0])

        self.filter_coeffs = (b, a)
        return b, a

    def apply_filter(
        self, signal_data: np.ndarray, b: np.ndarray, a: np.ndarray
    ) -> np.ndarray:
        """Aplica filtro com verificação de estabilidade"""
        try:
            # Verificar estabilidade
            poles = np.roots(a)
            if np.any(np.abs(poles) >= 1):
                print("Aviso: Filtro instável detectado")

            # Aplicar filtro
            return signal.filtfilt(b, a, signal_data)
        except Exception as e:
            print(f"Erro ao aplicar filtro: {e}")
            return signal_data

    def compute_state_space(
        self, b: np.ndarray, a: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Converte função de transferência para espaço de estados"""
        return signal.tf2ss(b, a)

    def check_controllability(
        self, A: np.ndarray, B: np.ndarray
    ) -> Tuple[bool, int, np.ndarray]:
        """
        Verifica controlabilidade do sistema
        Retorna: (é_controlável, rank, matriz_controlabilidade)
        """
        n = A.shape[0]

        # Construir matriz de controlabilidade
        C = B.reshape(-1, 1) if B.ndim == 1 else B

        for i in range(1, n):
            C = np.hstack([C, np.linalg.matrix_power(A, i) @ B.reshape(-1, 1)])

        # Calcular rank
        rank = np.linalg.matrix_rank(C)
        is_controllable = rank == n

        return is_controllable, rank, C

    def check_observability(
        self, A: np.ndarray, C: np.ndarray
    ) -> Tuple[bool, int, np.ndarray]:
        """
        Verifica observabilidade do sistema
        Retorna: (é_observável, rank, matriz_observabilidade)
        """
        n = A.shape[0]

        # Construir matriz de observabilidade
        obs_mat = C.reshape(1, -1) if C.ndim == 1 else C

        for i in range(1, n):
            obs_mat = np.vstack(
                [obs_mat, C.reshape(1, -1) @ np.linalg.matrix_power(A, i)]
            )

        # Calcular rank
        rank = np.linalg.matrix_rank(obs_mat)
        is_observable = rank == n

        return is_observable, rank, obs_mat

    def check_stability(
        self, poles: np.ndarray, discrete: bool = True
    ) -> Tuple[bool, str]:
        """
        Verifica estabilidade BIBO do sistema
        Retorna: (é_estável, descrição)
        """
        if discrete:
            # Sistema discreto: |pólos| < 1
            max_magnitude = np.max(np.abs(poles)) if len(poles) > 0 else 0
            is_stable = max_magnitude < 1

            if is_stable:
                margin = 1 - max_magnitude
                desc = "ESTÁVEL: Todos os pólos dentro do círculo unitário\n"
                desc += f"Margem de estabilidade: {margin:.3f}"
            else:
                unstable_poles = poles[np.abs(poles) >= 1]
                desc = f"INSTÁVEL: {len(unstable_poles)} pólo(s) fora do círculo unitário\n"
                desc += f"Maior magnitude: {max_magnitude:.3f}"
        else:
            # Sistema contínuo: Re(pólos) < 0
            max_real = np.max(np.real(poles)) if len(poles) > 0 else -1
            is_stable = max_real < 0

            if is_stable:
                desc = "ESTÁVEL: Todos os pólos no semiplano esquerdo\n"
                desc += f"Margem de estabilidade: {-max_real:.3f}"
            else:
                unstable_poles = poles[np.real(poles) >= 0]
                desc = f"INSTÁVEL: {len(unstable_poles)} pólo(s) no semiplano direito\n"
                desc += f"Maior parte real: {max_real:.3f}"

        return is_stable, desc

    def compute_system_response(
        self,
        b: np.ndarray,
        a: np.ndarray,
        input_type: str = "impulse",
        n_samples: int = 100,
    ):
        """Calcula resposta do sistema a diferentes entradas"""
        if input_type == "impulse":
            x = np.zeros(n_samples)
            x[0] = 1
        elif input_type == "step":
            x = np.ones(n_samples)
        elif input_type == "ramp":
            x = np.arange(n_samples)
        else:
            x = np.random.randn(n_samples)

        y = signal.lfilter(b, a, x)
        return x, y


# ==================== Interface Principal ====================


class ModernSignalSystemsApp(ctk.CTk):
    """Aplicação principal com análises completas"""

    def __init__(self):
        super().__init__()

        # Configurações da janela
        self.title("Visualizador Avançado de Sinais e Sistemas II")
        self.geometry("1600x900")
        self.center_window()

        # Componentes
        self.processor = SignalProcessor()
        self.audio_player = AudioPlayer()

        # Variáveis de estado
        self.current_signal_type = SignalType.SINE
        self.current_signal = np.array([])
        self.current_time = np.array([])
        self.fs = 8000
        self.is_updating = False

        # Construir interface
        self.setup_ui()

        # Gerar sinal inicial
        self.generate_signal()

    def center_window(self):
        """Centraliza janela na tela"""
        self.update_idletasks()
        width = 1600
        height = 900
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def setup_ui(self):
        """Constrói interface completa"""
        # Layout principal
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Painel lateral
        self.setup_sidebar()

        # Área de visualização
        self.setup_visualization_area()

    def setup_sidebar(self):
        """Configura painel lateral com controles"""
        # Frame principal do sidebar
        self.sidebar = ctk.CTkScrollableFrame(self, width=380, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        # Título
        title = ctk.CTkLabel(
            self.sidebar,
            text="SINAIS E SISTEMAS II",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 5))

        subtitle = ctk.CTkLabel(
            self.sidebar,
            text="Análise Completa de Sistemas Digitais",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        subtitle.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 20))

        # === GERAÇÃO DE SINAIS ===
        self.create_section("📊 GERAÇÃO DE SINAIS", 2)

        # Tipo de sinal
        ctk.CTkLabel(self.sidebar, text="Tipo de Sinal:").grid(
            row=3, column=0, padx=20, pady=5, sticky="w"
        )
        self.signal_type_var = ctk.StringVar(value=SignalType.SINE.value[0])
        self.signal_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=[s.value[0] for s in SignalType if s != SignalType.CUSTOM],
            variable=self.signal_type_var,
            command=self.on_signal_type_change,
            width=340,
        )
        self.signal_menu.grid(row=4, column=0, columnspan=2, padx=20, pady=5)

        # Descrição
        self.signal_desc = ctk.CTkLabel(
            self.sidebar,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#7dd3fc",
            wraplength=340,
            justify="left",
        )
        self.signal_desc.grid(row=5, column=0, columnspan=2, padx=20, pady=(0, 10))

        # Frequência
        self.freq_label = ctk.CTkLabel(self.sidebar, text="Frequência: 440 Hz")
        self.freq_label.grid(row=6, column=0, padx=20, pady=5, sticky="w")
        self.freq_slider = ctk.CTkSlider(
            self.sidebar, from_=1, to=2000, command=self.on_freq_change, width=340
        )
        self.freq_slider.set(440)
        self.freq_slider.grid(row=7, column=0, columnspan=2, padx=20, pady=5)

        # Amplitude
        self.amp_label = ctk.CTkLabel(self.sidebar, text="Amplitude: 1.00")
        self.amp_label.grid(row=8, column=0, padx=20, pady=5, sticky="w")
        self.amp_slider = ctk.CTkSlider(
            self.sidebar, from_=0, to=2, command=self.on_amp_change, width=340
        )
        self.amp_slider.set(1.0)
        self.amp_slider.grid(row=9, column=0, columnspan=2, padx=20, pady=5)

        # Taxa de amostragem
        self.fs_label = ctk.CTkLabel(self.sidebar, text="Taxa de Amostragem: 8000 Hz")
        self.fs_label.grid(row=10, column=0, padx=20, pady=5, sticky="w")
        self.fs_slider = ctk.CTkSlider(
            self.sidebar, from_=1000, to=48000, command=self.on_fs_change, width=340
        )
        self.fs_slider.set(8000)
        self.fs_slider.grid(row=11, column=0, columnspan=2, padx=20, pady=5)

        # Botões de controle de áudio
        audio_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        audio_frame.grid(row=12, column=0, columnspan=2, padx=20, pady=10)

        self.play_btn = ctk.CTkButton(
            audio_frame,
            text="▶ Reproduzir",
            command=self.play_audio,
            width=110,
            fg_color="#10b981",
            hover_color="#059669",
        )
        self.play_btn.pack(side="left", padx=5)

        self.stop_btn = ctk.CTkButton(
            audio_frame,
            text="⏹ Parar",
            command=self.stop_audio,
            width=110,
            fg_color="#ef4444",
            hover_color="#dc2626",
        )
        self.stop_btn.pack(side="left", padx=5)

        self.gen_btn = ctk.CTkButton(
            audio_frame,
            text="🔄 Gerar",
            command=self.generate_signal,
            width=110,
            fg_color="#3b82f6",
            hover_color="#2563eb",
        )
        self.gen_btn.pack(side="left", padx=5)

        # === PROCESSAMENTO ===
        self.create_section("🎛️ PROCESSAMENTO", 13)

        # Ruído
        self.noise_label = ctk.CTkLabel(self.sidebar, text="SNR: 20 dB")
        self.noise_label.grid(row=14, column=0, padx=20, pady=5, sticky="w")
        self.noise_slider = ctk.CTkSlider(
            self.sidebar, from_=0, to=50, command=self.on_noise_change, width=340
        )
        self.noise_slider.set(20)
        self.noise_slider.grid(row=15, column=0, columnspan=2, padx=20, pady=5)

        # === FILTROS ===
        self.create_section("🔧 FILTROS DIGITAIS", 16)

        # Tipo de filtro
        self.filter_type_var = ctk.StringVar(value="Passa-Baixa")
        filter_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["Passa-Baixa", "Passa-Alta", "Passa-Banda", "Rejeita-Banda"],
            variable=self.filter_type_var,
            command=self.on_filter_change,
            width=340,
        )
        filter_menu.grid(row=17, column=0, columnspan=2, padx=20, pady=5)

        # Ordem do filtro
        self.order_label = ctk.CTkLabel(self.sidebar, text="Ordem: 4")
        self.order_label.grid(row=18, column=0, padx=20, pady=5, sticky="w")
        self.order_slider = ctk.CTkSlider(
            self.sidebar,
            from_=1,
            to=10,
            command=self.on_order_change,
            width=340,
            number_of_steps=9,
        )
        self.order_slider.set(4)
        self.order_slider.grid(row=19, column=0, columnspan=2, padx=20, pady=5)

        # Frequência de corte
        self.cutoff_label = ctk.CTkLabel(
            self.sidebar, text="Frequência de Corte: 1000 Hz"
        )
        self.cutoff_label.grid(row=20, column=0, padx=20, pady=5, sticky="w")
        self.cutoff_slider = ctk.CTkSlider(
            self.sidebar, from_=10, to=4000, command=self.on_cutoff_change, width=340
        )
        self.cutoff_slider.set(1000)
        self.cutoff_slider.grid(row=21, column=0, columnspan=2, padx=20, pady=5)

        # Aplicar filtro
        apply_btn = ctk.CTkButton(
            self.sidebar,
            text="Aplicar Filtro",
            command=self.apply_filter,
            width=340,
            fg_color="#8b5cf6",
            hover_color="#7c3aed",
        )
        apply_btn.grid(row=22, column=0, columnspan=2, padx=20, pady=10)

        # === ANÁLISE AVANÇADA ===
        self.create_section("🔬 ANÁLISE DO SISTEMA", 23)

        # Botões de análise
        analysis_buttons = [
            ("📊 Análise de Estabilidade", self.analyze_stability),
            ("🎯 Análise de Controlabilidade", self.analyze_controllability),
            ("👁️ Análise de Observabilidade", self.analyze_observability),
            ("⚡ Resposta ao Impulso", self.impulse_response),
            ("📈 Resposta ao Degrau", self.step_response),
        ]

        for i, (text, command) in enumerate(analysis_buttons):
            btn = ctk.CTkButton(
                self.sidebar,
                text=text,
                command=command,
                width=340,
                height=35,
                fg_color="#64748b",
                hover_color="#475569",
            )
            btn.grid(row=24 + i, column=0, columnspan=2, padx=20, pady=3)

        # Carregar áudio
        load_btn = ctk.CTkButton(
            self.sidebar,
            text="📁 Carregar Áudio WAV",
            command=self.load_audio,
            width=340,
            height=40,
            fg_color="#0ea5e9",
            hover_color="#0284c7",
        )
        load_btn.grid(row=30, column=0, columnspan=2, padx=20, pady=(20, 10))

    def create_section(self, title: str, row: int):
        """Cria seção no sidebar"""
        header = ctk.CTkLabel(
            self.sidebar,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#60a5fa",
        )
        header.grid(row=row, column=0, columnspan=2, padx=20, pady=(15, 5), sticky="w")

    def setup_visualization_area(self):
        """Configura área de visualização"""
        # Frame principal
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Notebook com abas
        self.notebook = ctk.CTkTabview(self.main_frame, corner_radius=10)
        self.notebook.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Criar abas
        self.tab_time = self.notebook.add("⏱️ Tempo")
        self.tab_freq = self.notebook.add("📊 Frequência")
        self.tab_spec = self.notebook.add("🌈 Espectrograma")
        self.tab_pz = self.notebook.add("🎯 Polos/Zeros")
        self.tab_filter = self.notebook.add("🔧 Filtros")
        self.tab_analysis = self.notebook.add("📈 Análise")
        self.tab_fourier = self.notebook.add("📐 TF Fourier")  # Nova aba didática
        # Configurar plots
        self.setup_plots()
        # Estado para animação Fourier
        self.fourier_anim_running = False
        self.fourier_scan_mode = False
        self.fourier_freq = 440
        self.fourier_step_ms = 30
        self.fourier_spectrum_data = []

    def setup_plots(self):
        """Configura todos os plots"""
        # Plot do domínio do tempo
        self.fig_time = Figure(figsize=(10, 6), facecolor="#212121")
        self.ax_time = self.fig_time.add_subplot(111, facecolor="#1a1a1a")
        self.canvas_time = FigureCanvasTkAgg(self.fig_time, self.tab_time)
        self.canvas_time.get_tk_widget().pack(fill="both", expand=True)

        # Plot do domínio da frequência
        self.fig_freq = Figure(figsize=(10, 6), facecolor="#212121")
        self.ax_mag = self.fig_freq.add_subplot(211, facecolor="#1a1a1a")
        self.ax_phase = self.fig_freq.add_subplot(212, facecolor="#1a1a1a")
        self.fig_freq.tight_layout()
        self.canvas_freq = FigureCanvasTkAgg(self.fig_freq, self.tab_freq)
        self.canvas_freq.get_tk_widget().pack(fill="both", expand=True)

        # Espectrograma
        self.fig_spec = Figure(figsize=(10, 6), facecolor="#212121")
        self.ax_spec = self.fig_spec.add_subplot(111, facecolor="#1a1a1a")
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, self.tab_spec)
        self.canvas_spec.get_tk_widget().pack(fill="both", expand=True)

        # Diagrama de polos e zeros
        self.fig_pz = Figure(figsize=(10, 6), facecolor="#212121")
        self.ax_pz = self.fig_pz.add_subplot(121, facecolor="#1a1a1a")
        self.ax_freq_resp = self.fig_pz.add_subplot(122, facecolor="#1a1a1a")
        self.fig_pz.tight_layout()
        self.canvas_pz = FigureCanvasTkAgg(self.fig_pz, self.tab_pz)
        self.canvas_pz.get_tk_widget().pack(fill="both", expand=True)

        # Análise de filtros
        self.fig_filter = Figure(figsize=(10, 6), facecolor="#212121")
        self.ax_filter_mag = self.fig_filter.add_subplot(221, facecolor="#1a1a1a")
        self.ax_filter_phase = self.fig_filter.add_subplot(222, facecolor="#1a1a1a")
        self.ax_filter_impulse = self.fig_filter.add_subplot(223, facecolor="#1a1a1a")
        self.ax_filter_step = self.fig_filter.add_subplot(224, facecolor="#1a1a1a")
        self.fig_filter.tight_layout()
        self.canvas_filter = FigureCanvasTkAgg(self.fig_filter, self.tab_filter)
        self.canvas_filter.get_tk_widget().pack(fill="both", expand=True)

        # Análise do sistema
        self.fig_analysis = Figure(figsize=(10, 6), facecolor="#212121")
        self.ax_analysis = self.fig_analysis.add_subplot(111, facecolor="#1a1a1a")
        self.canvas_analysis = FigureCanvasTkAgg(self.fig_analysis, self.tab_analysis)
        self.canvas_analysis.get_tk_widget().pack(fill="both", expand=True)

        # Área de texto para resultados
        self.analysis_text = ctk.CTkTextbox(
            self.tab_analysis,
            width=500,
            height=200,
            font=ctk.CTkFont(family="Courier", size=12),
        )
        self.analysis_text.pack(side="bottom", fill="x", padx=10, pady=10)

        # Figura da animação Fourier (quatro subplots)
        if hasattr(self, "tab_fourier"):
            self.fig_fourier = Figure(figsize=(10, 6), facecolor="#212121")
            gs = self.fig_fourier.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
            self.ax_fourier_time = self.fig_fourier.add_subplot(
                gs[0, 0], facecolor="#1a1a1a"
            )
            self.ax_fourier_circle = self.fig_fourier.add_subplot(
                gs[0, 1], facecolor="#1a1a1a"
            )
            self.ax_fourier_accum = self.fig_fourier.add_subplot(
                gs[1, 0], facecolor="#1a1a1a"
            )
            self.ax_fourier_spectrum = self.fig_fourier.add_subplot(
                gs[1, 1], facecolor="#1a1a1a"
            )
            self.canvas_fourier = FigureCanvasTkAgg(self.fig_fourier, self.tab_fourier)
            self.canvas_fourier.get_tk_widget().pack(fill="both", expand=True)

            # Controles aprimorados
            ctrl_frame = ctk.CTkFrame(self.tab_fourier)
            ctrl_frame.pack(fill="x", padx=10, pady=5)

            # Linha 1: Frequência e Timeline
            freq_line = ctk.CTkFrame(ctrl_frame)
            freq_line.pack(fill="x", pady=2)

            self.fourier_freq_label = ctk.CTkLabel(
                freq_line, text="Frequência: 440 Hz", width=120
            )
            self.fourier_freq_label.pack(side="left", padx=5)

            self.fourier_freq_slider = ctk.CTkSlider(
                freq_line,
                from_=10,
                to=2000,
                width=250,
                command=self.on_fourier_freq_change,
            )
            self.fourier_freq_slider.set(440)
            self.fourier_freq_slider.pack(side="left", padx=5)

            # Timeline slider
            self.timeline_label = ctk.CTkLabel(freq_line, text="Posição: 0%", width=100)
            self.timeline_label.pack(side="left", padx=(20, 5))

            self.timeline_slider = ctk.CTkSlider(
                freq_line, from_=0, to=100, width=200, command=self.on_timeline_change
            )
            self.timeline_slider.set(0)
            self.timeline_slider.pack(side="left", padx=5)

            # Linha 2: Controles de reprodução
            play_line = ctk.CTkFrame(ctrl_frame)
            play_line.pack(fill="x", pady=2)

            self.btn_fourier_play = ctk.CTkButton(
                play_line,
                text="▶ Anima",
                width=90,
                command=self.toggle_fourier_animation,
                fg_color="#10b981",
                hover_color="#059669",
            )
            self.btn_fourier_play.pack(side="left", padx=5)

            self.btn_fourier_scan = ctk.CTkButton(
                play_line,
                text="🔍 Varredura",
                width=110,
                command=self.toggle_fourier_scan,
                fg_color="#6366f1",
                hover_color="#4f46e5",
            )
            self.btn_fourier_scan.pack(side="left", padx=5)

            # Seletor de velocidade
            self.speed_label = ctk.CTkLabel(play_line, text="Velocidade:", width=80)
            self.speed_label.pack(side="left", padx=(20, 5))

            self.speed_var = ctk.StringVar(value="1x")
            self.speed_menu = ctk.CTkOptionMenu(
                play_line,
                values=["0.5x", "1x", "2x", "3x", "Max"],
                variable=self.speed_var,
                command=self.on_speed_change,
                width=80,
            )
            self.speed_menu.pack(side="left", padx=5)

            # Reset button
            self.btn_reset = ctk.CTkButton(
                play_line,
                text="⏮ Reset",
                width=80,
                command=self.reset_fourier_animation,
                fg_color="#94a3b8",
                hover_color="#64748b",
            )
            self.btn_reset.pack(side="left", padx=5)

            # Info text
            info = "Visualização interativa da Transformada de Fourier: soma de fasores para formar o espectro."
            ctk.CTkLabel(
                self.tab_fourier,
                text=info,
                wraplength=1100,
                justify="left",
                text_color="#94a3b8",
            ).pack(fill="x", padx=15, pady=(0, 10))

            # Inicializar variáveis de animação
            self.fourier_thread = None
            self.fourier_lock = threading.Lock()
            self.animation_speed = 1.0
            self.timeline_position = 0
            self.fourier_cache = {}
            self.init_fourier_axes()
        # Placeholder para colorbar do espectrograma (para não acumular)
        self.spec_colorbar = None

    def setup_axis_style(self, ax, xlabel="", ylabel="", title=""):
        """Estiliza os eixos"""
        ax.set_xlabel(xlabel, color="#e2e8f0", fontsize=10)
        ax.set_ylabel(ylabel, color="#e2e8f0", fontsize=10)
        ax.set_title(title, color="#f1f5f9", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2, color="#475569")
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#475569")
            spine.set_linewidth(0.5)

    # ==================== Callbacks ====================

    def on_signal_type_change(self, choice):
        """Mudança do tipo de sinal"""
        for signal_type in SignalType:
            if signal_type.value[0] == choice:
                self.current_signal_type = signal_type
                self.signal_desc.configure(text=signal_type.value[1])

                # Ajustar parâmetros ideais
                self.freq_slider.set(signal_type.value[2])

                # Desabilitar frequência para alguns sinais
                if signal_type in [
                    SignalType.IMPULSE,
                    SignalType.STEP,
                    SignalType.NOISE,
                ]:
                    self.freq_slider.configure(state="disabled")
                else:
                    self.freq_slider.configure(state="normal")
                break

        self.generate_signal()

    def on_freq_change(self, value):
        self.freq_label.configure(text=f"Frequência: {value:.0f} Hz")
        if not self.is_updating:
            self.generate_signal()

    def on_amp_change(self, value):
        self.amp_label.configure(text=f"Amplitude: {value:.2f}")
        if not self.is_updating:
            self.generate_signal()

    def on_fs_change(self, value):
        self.fs = int(value)
        self.fs_label.configure(text=f"Taxa de Amostragem: {self.fs} Hz")
        if not self.is_updating:
            self.generate_signal()

    def on_noise_change(self, value):
        self.noise_label.configure(text=f"SNR: {value:.0f} dB")
        if not self.is_updating:
            self.generate_signal()

    def on_filter_change(self, choice):
        self.update_filter_response()

    def on_order_change(self, value):
        self.order_label.configure(text=f"Ordem: {int(value)}")
        self.update_filter_response()

    def on_cutoff_change(self, value):
        self.cutoff_label.configure(text=f"Frequência de Corte: {value:.0f} Hz")
        self.update_filter_response()

    # ==================== Geração e Processamento ====================

    def generate_signal(self):
        """Gera novo sinal com parâmetros atuais"""
        if self.is_updating:
            return
        self.is_updating = True

        try:
            # Gerar sinal base
            duration = self.current_signal_type.value[3]
            self.current_time, self.current_signal = self.processor.generate_signal(
                self.current_signal_type,
                self.fs,
                duration,
                self.freq_slider.get(),
                self.amp_slider.get(),
            )

            # Adicionar ruído se necessário
            snr = self.noise_slider.get()
            if snr < 50 and len(self.current_signal) > 0:
                signal_power = np.mean(self.current_signal**2)
                if signal_power > 0:
                    noise_power = signal_power / (10 ** (snr / 10))
                    noise = np.sqrt(noise_power) * np.random.randn(
                        len(self.current_signal)
                    )
                    self.current_signal = self.current_signal + noise

            # Atualizar visualizações
            self.update_all_plots()

        finally:
            self.is_updating = False

    def apply_filter(self):
        """Aplica filtro ao sinal atual"""
        if len(self.current_signal) == 0:
            return
        b, a = self.processor.design_filter(
            self.filter_type_var.get(),
            self.fs,
            int(self.order_slider.get()),
            self.cutoff_slider.get(),
        )
        filtered = self.processor.apply_filter(self.current_signal, b, a)
        # Guardar original para comparação
        self.last_unfiltered_signal = self.current_signal.copy()
        self.current_signal = filtered
        # Atualizar plots
        self.ax_time.clear()
        self.ax_time.plot(
            self.current_time,
            self.last_unfiltered_signal,
            "c-",
            alpha=0.5,
            label="Original",
        )
        self.ax_time.plot(
            self.current_time,
            self.current_signal,
            "m-",
            linewidth=1.8,
            label="Filtrado",
        )
        self.ax_time.legend()
        self.setup_axis_style(self.ax_time, "Tempo (s)", "Amplitude", "Filtro Aplicado")
        self.canvas_time.draw()
        self.update_frequency_plot()
        self.update_spectrogram()
        self.filtered_signal = filtered

    def update_all_plots(self):
        """Atualiza todas as visualizações"""
        self.update_time_plot()
        self.update_frequency_plot()
        self.update_spectrogram()
        self.update_filter_response()

    def update_time_plot(self):
        """Atualiza plot do domínio do tempo"""
        if len(self.current_signal) == 0:
            return

        self.ax_time.clear()
        self.ax_time.plot(self.current_time, self.current_signal, "b-", linewidth=1.5)

        # Adicionar estatísticas
        mean = np.mean(self.current_signal)
        std = np.std(self.current_signal)
        rms = np.sqrt(np.mean(self.current_signal**2))

        stats_text = f"μ={mean:.3f} | σ={std:.3f} | RMS={rms:.3f}"
        self.ax_time.text(
            0.02,
            0.98,
            stats_text,
            transform=self.ax_time.transAxes,
            fontsize=9,
            color="#94a3b8",
            va="top",
            bbox=dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.8),
        )

        self.setup_axis_style(
            self.ax_time,
            "Tempo (s)",
            "Amplitude",
            f"Sinal {self.current_signal_type.value[0]}",
        )
        self.canvas_time.draw()

    def update_frequency_plot(self):
        """Atualiza plot do domínio da frequência"""
        if len(self.current_signal) == 0:
            return

        freq, mag, phase = self.processor.compute_fft(self.current_signal, self.fs)

        # Magnitude
        self.ax_mag.clear()
        self.ax_mag.semilogy(freq, mag + 1e-10, "g-", linewidth=1.5)

        # Marcar picos
        if len(mag) > 0:
            peaks, _ = signal.find_peaks(mag, height=np.max(mag) * 0.1)
            if len(peaks) > 0:
                for peak in peaks[:5]:  # Mostrar até 5 picos
                    self.ax_mag.axvline(
                        freq[peak], color="r", linestyle="--", alpha=0.5
                    )
                    self.ax_mag.text(
                        freq[peak],
                        mag[peak],
                        f"{freq[peak]:.0f}Hz",
                        fontsize=8,
                        color="#f87171",
                    )

        self.setup_axis_style(
            self.ax_mag, "Frequência (Hz)", "Magnitude", "Espectro de Magnitude"
        )

        # Fase
        self.ax_phase.clear()
        self.ax_phase.plot(freq, np.unwrap(phase), "orange", linewidth=1.5)
        self.setup_axis_style(
            self.ax_phase, "Frequência (Hz)", "Fase (rad)", "Espectro de Fase"
        )

        self.fig_freq.tight_layout()
        self.canvas_freq.draw()

    def update_spectrogram(self):
        """Atualiza espectrograma com correção"""
        # Encontrar blocos calculados previamente
        # (Reexecutar cálculo completo original)
        if len(self.current_signal) < 64:
            self.ax_spec.clear()
            self.ax_spec.text(
                0.5,
                0.5,
                "Sinal muito curto para espectrograma",
                ha="center",
                va="center",
                transform=self.ax_spec.transAxes,
                color="#94a3b8",
                fontsize=12,
            )
            self.setup_axis_style(
                self.ax_spec, "Tempo (s)", "Frequência (Hz)", "Espectrograma"
            )
            self.canvas_spec.draw()
            return

        # Calcular espectrograma
        f, t, Sxx = self.processor.compute_spectrogram(self.current_signal, self.fs)

        self.ax_spec.clear()

        # Plot com níveis adequados
        vmin = np.percentile(Sxx, 10)
        vmax = np.percentile(Sxx, 99)

        im = self.ax_spec.pcolormesh(
            t, f, Sxx, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax
        )

        self.ax_spec.set_ylim([0, self.fs / 2])
        self.setup_axis_style(
            self.ax_spec, "Tempo (s)", "Frequência (Hz)", "Espectrograma"
        )
        if self.spec_colorbar is not None:
            try:
                self.spec_colorbar.remove()
            except Exception:
                pass
            self.spec_colorbar = None
        self.spec_colorbar = self.fig_spec.colorbar(
            im, ax=self.ax_spec, label="Potência (dB)"
        )
        self.fig_spec.tight_layout()
        self.canvas_spec.draw()

    def update_filter_response(self):
        """Atualiza resposta do filtro com análise completa"""
        # Projetar filtro
        b, a = self.processor.design_filter(
            self.filter_type_var.get(),
            self.fs,
            int(self.order_slider.get()),
            self.cutoff_slider.get(),
        )

        # Resposta em frequência
        w, h = signal.freqz(b, a, worN=512)
        freq_hz = w * self.fs / (2 * np.pi)

        # Magnitude
        self.ax_filter_mag.clear()
        mag_db = 20 * np.log10(np.abs(h) + 1e-10)
        self.ax_filter_mag.plot(freq_hz, mag_db, "b-", linewidth=2)
        self.ax_filter_mag.axhline(-3, color="r", linestyle="--", alpha=0.5)
        self.ax_filter_mag.axvline(
            self.cutoff_slider.get(), color="g", linestyle="--", alpha=0.5
        )
        self.setup_axis_style(
            self.ax_filter_mag, "Frequência (Hz)", "Magnitude (dB)", ""
        )

        # Fase
        self.ax_filter_phase.clear()
        phase_deg = np.angle(h, deg=True)
        self.ax_filter_phase.plot(freq_hz, phase_deg, "orange", linewidth=2)
        self.setup_axis_style(self.ax_filter_phase, "Frequência (Hz)", "Fase (°)", "")

        # Resposta ao impulso
        self.ax_filter_impulse.clear()
        impulse, impulse_resp = self.processor.compute_system_response(
            b, a, "impulse", 50
        )
        self.ax_filter_impulse.stem(impulse_resp, basefmt=" ")
        self.setup_axis_style(
            self.ax_filter_impulse, "Amostras", "h[n]", "Resposta ao Impulso"
        )

        # Resposta ao degrau
        self.ax_filter_step.clear()
        step, step_resp = self.processor.compute_system_response(b, a, "step", 50)
        self.ax_filter_step.plot(step_resp, "g-", linewidth=2)
        self.setup_axis_style(
            self.ax_filter_step, "Amostras", "s[n]", "Resposta ao Degrau"
        )

        self.fig_filter.tight_layout()
        self.canvas_filter.draw()

        # Atualizar diagrama de polos e zeros
        self.update_pz_diagram(b, a)

    def update_pz_diagram(self, b, a):
        """Atualiza diagrama de polos e zeros"""
        zeros = np.roots(b)
        poles = np.roots(a)

        self.ax_pz.clear()

        # Círculo unitário
        theta = np.linspace(0, 2 * np.pi, 100)
        self.ax_pz.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3)

        # Plotar zeros e polos
        if len(zeros) > 0:
            self.ax_pz.scatter(
                np.real(zeros),
                np.imag(zeros),
                s=80,
                c="blue",
                marker="o",
                edgecolors="white",
                linewidths=2,
                label="Zeros",
            )
        if len(poles) > 0:
            self.ax_pz.scatter(
                np.real(poles),
                np.imag(poles),
                s=100,
                c="red",
                marker="x",
                linewidths=3,
                label="Pólos",
            )

        # Verificar estabilidade
        is_stable, desc = self.processor.check_stability(poles)
        color = "green" if is_stable else "red"
        self.ax_pz.text(
            0.02,
            0.98,
            desc.split("\n")[0],
            transform=self.ax_pz.transAxes,
            fontsize=10,
            color=color,
            fontweight="bold",
            va="top",
        )

        self.ax_pz.set_xlim([-2, 2])
        self.ax_pz.set_ylim([-2, 2])
        self.ax_pz.legend()
        self.ax_pz.set_aspect("equal")
        self.setup_axis_style(
            self.ax_pz, "Real", "Imaginário", "Diagrama de Pólos e Zeros"
        )

        # Resposta em frequência no segundo subplot
        w, h = signal.freqz(b, a, worN=512)
        self.ax_freq_resp.clear()
        self.ax_freq_resp.plot(
            w / np.pi, 20 * np.log10(np.abs(h)), "purple", linewidth=2
        )
        self.setup_axis_style(
            self.ax_freq_resp,
            "Frequência Normalizada (×π rad/sample)",
            "Magnitude (dB)",
            "Resposta em Frequência",
        )

        self.fig_pz.tight_layout()
        self.canvas_pz.draw()

    # ==================== Análises Avançadas ====================

    def analyze_stability(self):
        """Análise completa de estabilidade"""
        if self.processor.filter_coeffs is None:
            self.show_message("Erro", "Projete um filtro primeiro!")
            return

        b, a = self.processor.filter_coeffs
        poles = np.roots(a)
        zeros = np.roots(b)

        is_stable, desc = self.processor.check_stability(poles)

        # Preparar relatório
        report = "=" * 50 + "\n"
        report += "ANÁLISE DE ESTABILIDADE DO SISTEMA\n"
        report += "=" * 50 + "\n\n"

        report += f"Tipo de Filtro: {self.filter_type_var.get()}\n"
        report += f"Ordem: {int(self.order_slider.get())}\n"
        report += f"Frequência de Corte: {self.cutoff_slider.get():.0f} Hz\n\n"

        report += "PÓLOS DO SISTEMA:\n"
        report += "-" * 30 + "\n"
        for i, pole in enumerate(poles):
            mag = np.abs(pole)
            angle = np.angle(pole, deg=True)
            report += f"p{i+1} = {pole:.4f}\n"
            report += f"  |p{i+1}| = {mag:.4f} {'✓' if mag < 1 else '✗'}\n"
            report += f"  ∠p{i+1} = {angle:.2f}°\n\n"

        report += "ZEROS DO SISTEMA:\n"
        report += "-" * 30 + "\n"
        for i, zero in enumerate(zeros):
            report += f"z{i+1} = {zero:.4f}\n"

        report += "\n" + "=" * 50 + "\n"
        report += f"CONCLUSÃO: {desc}\n"
        report += "=" * 50 + "\n"

        if is_stable:
            report += "\n✅ Sistema BIBO estável"
            report += "\nTodos os pólos estão dentro do círculo unitário"
        else:
            report += "\n❌ Sistema INSTÁVEL"
            report += "\nExistem pólos fora do círculo unitário"

        # Mostrar no campo de texto
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", report)

        # Mudar para aba de análise
        self.notebook.set("📈 Análise")

    def analyze_controllability(self):
        """Análise completa de controlabilidade"""
        if self.processor.filter_coeffs is None:
            self.show_message("Erro", "Projete um filtro primeiro!")
            return

        b, a = self.processor.filter_coeffs
        A, B, C, D = self.processor.compute_state_space(b, a)

        is_controllable, rank, ctrl_matrix = self.processor.check_controllability(A, B)

        # Relatório
        report = "=" * 50 + "\n"
        report += "ANÁLISE DE CONTROLABILIDADE\n"
        report += "=" * 50 + "\n\n"

        report += f"Dimensão do sistema: {A.shape[0]}\n"
        report += f"Rank da matriz de controlabilidade: {rank}\n"
        report += f"Rank necessário para controlabilidade: {A.shape[0]}\n\n"

        report += "MATRIZ A (Estado):\n"
        report += str(A) + "\n\n"

        report += "MATRIZ B (Entrada):\n"
        report += str(B) + "\n\n"

        report += "MATRIZ DE CONTROLABILIDADE [B|AB|A²B|...]:\n"
        report += str(ctrl_matrix) + "\n\n"

        # Valores singulares para análise de condicionamento
        U, s, Vt = svd(ctrl_matrix)
        report += "VALORES SINGULARES:\n"
        for i, sv in enumerate(s):
            report += f"σ{i+1} = {sv:.6f}\n"

        condition_number = s[0] / (s[-1] + 1e-10)
        report += f"\nNúmero de Condição: {condition_number:.2f}\n"

        report += "\n" + "=" * 50 + "\n"
        if is_controllable:
            report += "✅ SISTEMA COMPLETAMENTE CONTROLÁVEL\n"
            report += "É possível levar o sistema de qualquer estado inicial\n"
            report += "para qualquer estado final em tempo finito."
        else:
            report += "❌ SISTEMA NÃO É COMPLETAMENTE CONTROLÁVEL\n"
            report += f"Existem {A.shape[0] - rank} estado(s) não controlável(is)\n"
            report += "Não é possível controlar completamente o sistema."

        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", report)
        self.notebook.set("📈 Análise")

    def analyze_observability(self):
        """Análise completa de observabilidade"""
        if self.processor.filter_coeffs is None:
            self.show_message("Erro", "Projete um filtro primeiro!")
            return

        b, a = self.processor.filter_coeffs
        A, B, C, D = self.processor.compute_state_space(b, a)

        is_observable, rank, obs_matrix = self.processor.check_observability(A, C)

        # Relatório
        report = "=" * 50 + "\n"
        report += "ANÁLISE DE OBSERVABILIDADE\n"
        report += "=" * 50 + "\n\n"

        report += f"Dimensão do sistema: {A.shape[0]}\n"
        report += f"Rank da matriz de observabilidade: {rank}\n"
        report += f"Rank necessário para observabilidade: {A.shape[0]}\n\n"

        report += "MATRIZ A (Estado):\n"
        report += str(A) + "\n\n"

        report += "MATRIZ C (Saída):\n"
        report += str(C) + "\n\n"

        report += "MATRIZ DE OBSERVABILIDADE [C; CA; CA²; ...]:\n"
        report += str(obs_matrix) + "\n\n"

        # Valores singulares
        U, s, Vt = svd(obs_matrix)
        report += "VALORES SINGULARES:\n"
        for i, sv in enumerate(s[: min(len(s), 10)]):
            report += f"σ{i+1} = {sv:.6f}\n"

        condition_number = s[0] / (s[-1] + 1e-10) if len(s) > 0 else np.inf
        report += f"\nNúmero de Condição: {condition_number:.2f}\n"

        report += "\n" + "=" * 50 + "\n"
        if is_observable:
            report += "✅ SISTEMA COMPLETAMENTE OBSERVÁVEL\n"
            report += "É possível determinar o estado interno completo\n"
            report += "a partir das medições de saída."
        else:
            report += "❌ SISTEMA NÃO É COMPLETAMENTE OBSERVÁVEL\n"
            report += f"Existem {A.shape[0] - rank} estado(s) não observável(is)\n"
            report += "Não é possível reconstruir completamente o estado interno."

        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", report)
        self.notebook.set("📈 Análise")

    def impulse_response(self):
        """Calcula e mostra resposta ao impulso"""
        if self.processor.filter_coeffs is None:
            self.show_message("Erro", "Projete um filtro primeiro!")
            return

        b, a = self.processor.filter_coeffs
        impulse, response = self.processor.compute_system_response(b, a, "impulse", 100)

        self.ax_analysis.clear()
        self.ax_analysis.stem(response, basefmt=" ")

        # Calcular métricas
        energy = np.sum(response**2)
        peak = np.max(np.abs(response))
        peak_idx = np.argmax(np.abs(response))

        info_text = f"Energia: {energy:.3f}\nPico: {peak:.3f} em n={peak_idx}"
        self.ax_analysis.text(
            0.02,
            0.98,
            info_text,
            transform=self.ax_analysis.transAxes,
            fontsize=10,
            color="#94a3b8",
            va="top",
            bbox=dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.8),
        )

        self.setup_axis_style(
            self.ax_analysis, "Amostras (n)", "h[n]", "Resposta ao Impulso do Sistema"
        )
        self.canvas_analysis.draw()
        self.notebook.set("📈 Análise")

    def step_response(self):
        """Calcula e mostra resposta ao degrau"""
        if self.processor.filter_coeffs is None:
            self.show_message("Erro", "Projete um filtro primeiro!")
            return

        b, a = self.processor.filter_coeffs
        step, response = self.processor.compute_system_response(b, a, "step", 100)

        self.ax_analysis.clear()
        self.ax_analysis.plot(response, "g-", linewidth=2)

        # Calcular métricas
        steady_state = response[-1]

        # Tempo de subida (10% a 90%)
        if steady_state != 0:
            t10 = np.where(response >= 0.1 * steady_state)[0]
            t90 = np.where(response >= 0.9 * steady_state)[0]
            rise_time = t90[0] - t10[0] if len(t10) > 0 and len(t90) > 0 else 0
        else:
            rise_time = 0

        # Overshoot
        overshoot = (np.max(response) - steady_state) / (steady_state + 1e-10) * 100

        info_text = f"Valor final: {steady_state:.3f}\nTempo de subida: {rise_time} amostras\nOvershoot: {overshoot:.1f}%"
        self.ax_analysis.text(
            0.02,
            0.98,
            info_text,
            transform=self.ax_analysis.transAxes,
            fontsize=10,
            color="#94a3b8",
            va="top",
            bbox=dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.8),
        )

        self.ax_analysis.axhline(steady_state, color="r", linestyle="--", alpha=0.5)
        self.setup_axis_style(
            self.ax_analysis, "Amostras (n)", "s[n]", "Resposta ao Degrau do Sistema"
        )
        self.canvas_analysis.draw()
        self.notebook.set("📈 Análise")

    # ==================== Funções de Áudio ====================

    def play_audio(self):
        """Reproduz o sinal atual como áudio (toggle)"""
        if len(self.current_signal) == 0:
            return
        if self.audio_player.playing:
            self.stop_audio()
            return
        data = getattr(self, "filtered_signal", self.current_signal)
        self.audio_player.play_signal(data, self.fs)
        self.play_btn.configure(text="⏹ Parar")

    def stop_audio(self):
        """Para a reprodução de áudio"""
        self.audio_player.stop()
        self.play_btn.configure(text="▶ Reproduzir")

    def load_audio(self):
        """Carrega arquivo de áudio WAV"""
        file_path = filedialog.askopenfilename(
            title="Selecionar arquivo WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )

        if file_path:
            try:
                fs, data = wavfile.read(file_path)

                # Converter para mono se necessário
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)

                # Normalizar
                data = data.astype(float) / (np.max(np.abs(data)) + 1e-10)

                # Limitar duração a 3 segundos
                max_samples = fs * 3
                if len(data) > max_samples:
                    data = data[:max_samples]

                # Atualizar
                self.current_signal = data
                self.current_time = np.arange(len(data)) / fs
                self.fs = fs
                self.current_signal_type = SignalType.CUSTOM

                # Atualizar controles
                self.fs_slider.set(fs)
                self.signal_desc.configure(text=f"Arquivo: {file_path.split('/')[-1]}")

                # Atualizar plots
                self.update_all_plots()

                self.show_message(
                    "Sucesso",
                    f"Áudio carregado!\nTaxa: {fs} Hz\nDuração: {len(data)/fs:.2f}s",
                )

            except Exception as e:
                self.show_message("Erro", f"Erro ao carregar áudio: {str(e)}")

    def show_message(self, title, message):
        """Mostra mensagem ao usuário"""
        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry("400x200")
        dialog.transient(self)
        dialog.grab_set()

        # Centralizar
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 200
        y = (dialog.winfo_screenheight() // 2) - 100
        dialog.geometry(f"+{x}+{y}")

        # Mensagem
        label = ctk.CTkLabel(dialog, text=message, wraplength=350)
        label.pack(pady=20)

        # Botão OK
        btn = ctk.CTkButton(dialog, text="OK", command=dialog.destroy)
        btn.pack(pady=10)

    # ==================== Animação de Fourier (Didática) ====================
    def init_fourier_axes(self):
        if not hasattr(self, "ax_fourier_time"):
            return
        self.ax_fourier_time.clear()
        self.ax_fourier_circle.clear()
        self.ax_fourier_accum.clear()
        self.ax_fourier_spectrum.clear()
        # Desenhar sinal se existir
        if len(getattr(self, "current_signal", [])) > 0:
            self.ax_fourier_time.plot(
                self.current_time, self.current_signal, color="#3b82f6"
            )
        self.setup_axis_style(
            self.ax_fourier_time, "Tempo (s)", "x[n]", "Sinal no Tempo"
        )
        self.setup_axis_style(
            self.ax_fourier_circle, "Real", "Imaginário", "Fasor e^{-j2πft}"
        )
        self.setup_axis_style(
            self.ax_fourier_accum, "Real", "Imaginário", "Soma Parcial"
        )
        self.setup_axis_style(
            self.ax_fourier_spectrum, "Frequência (Hz)", "|X(f)|", "Espectro Parcial"
        )
        theta = np.linspace(0, 2 * np.pi, 400)
        self.ax_fourier_circle.plot(
            np.cos(theta), np.sin(theta), color="#475569", alpha=0.35
        )
        self.ax_fourier_circle.set_aspect("equal")
        self.canvas_fourier.draw()

    def on_fourier_freq_change(self, val):
        self.fourier_freq = float(val)
        if hasattr(self, "fourier_freq_label"):
            self.fourier_freq_label.configure(
                text=f"Frequência: {self.fourier_freq:.0f} Hz"
            )
        # Se animando single-freq reinicia para refletir nova freq
        if getattr(self, "fourier_anim_running", False) and not getattr(
            self, "fourier_scan_mode", False
        ):
            self.start_fourier_animation(reset_spectrum=False)

    # ====== Métodos Fourier otimizados com threading ======
    def on_timeline_change(self, value):
        self.timeline_position = int(value)
        if hasattr(self, "timeline_label"):
            self.timeline_label.configure(text=f"Posição: {self.timeline_position}%")
        if not getattr(self, "fourier_anim_running", False):
            self.update_fourier_frame_at_position(self.timeline_position)

    def on_speed_change(self, choice):
        speeds = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0, "3x": 3.0, "Max": 0.01}
        self.animation_speed = speeds.get(choice, 1.0)
        self.fourier_step_ms = int(30 / self.animation_speed) if choice != "Max" else 1

    def reset_fourier_animation(self):
        self.stop_fourier_animation()
        if hasattr(self, "timeline_slider"):
            self.timeline_slider.set(0)
        self.timeline_position = 0
        self.fourier_spectrum_data = []
        self.init_fourier_axes()

    def stop_fourier_animation(self):
        self.fourier_anim_running = False
        self.fourier_scan_mode = False
        if getattr(self, "fourier_thread", None) and self.fourier_thread.is_alive():
            self.fourier_thread.join(timeout=0.5)
        if hasattr(self, "btn_fourier_play"):
            self.btn_fourier_play.configure(text="▶ Anima", fg_color="#10b981")
        if hasattr(self, "btn_fourier_scan"):
            self.btn_fourier_scan.configure(text="🔍 Varredura", fg_color="#6366f1")

    def toggle_fourier_animation(self):
        if not hasattr(self, "ax_fourier_time"):
            return
        if self.fourier_anim_running and not self.fourier_scan_mode:
            self.stop_fourier_animation()
        else:
            self.fourier_scan_mode = False
            self.fourier_anim_running = True
            if hasattr(self, "btn_fourier_play"):
                self.btn_fourier_play.configure(text="⏸ Pausa", fg_color="#dc2626")
            if hasattr(self, "btn_fourier_scan"):
                self.btn_fourier_scan.configure(text="🔍 Varredura", fg_color="#6366f1")
            self.fourier_thread = threading.Thread(
                target=self.run_fourier_animation, daemon=True
            )
            self.fourier_thread.start()

    def toggle_fourier_scan(self):
        if not hasattr(self, "ax_fourier_time"):
            return
        if self.fourier_scan_mode:
            self.stop_fourier_animation()
        else:
            self.fourier_scan_mode = True
            self.fourier_anim_running = True
            self.fourier_spectrum_data = []
            if hasattr(self, "btn_fourier_scan"):
                self.btn_fourier_scan.configure(text="⏸ Parar", fg_color="#dc2626")
            if hasattr(self, "btn_fourier_play"):
                self.btn_fourier_play.configure(text="▶ Anima", fg_color="#10b981")
            self.fourier_thread = threading.Thread(
                target=self.run_fourier_scan, daemon=True
            )
            self.fourier_thread.start()

    def run_fourier_animation(self):
        if len(getattr(self, "current_signal", [])) == 0:
            return
        N = len(self.current_signal)
        n = 0
        S = 0 + 0j
        self.after(0, self.setup_fourier_animation)
        time.sleep(0.05)
        # Preparar FFT completa para referência/linha alvo (uma vez)
        try:
            full_f, full_mag, _ = self.processor.compute_fft(
                self.current_signal, self.fs
            )
            self._full_fft_freq = full_f
            self._full_fft_mag = full_mag
        except Exception:
            self._full_fft_freq = np.array([])
            self._full_fft_mag = np.array([])

        # Magnitude alvo pela interpolação mais próxima
        def target_mag_at(freq):
            if self._full_fft_freq.size == 0:
                return None
            idx = np.argmin(np.abs(self._full_fft_freq - freq))
            return self._full_fft_mag[idx]

        target_mag = target_mag_at(self.fourier_freq)
        while self.fourier_anim_running and n < N:
            sample = self.current_signal[n]
            theta = 2 * np.pi * self.fourier_freq * n / self.fs
            phasor = np.exp(-1j * theta)
            S += sample * phasor
            progress = int(100 * n / N)
            self.after(
                0,
                lambda p=progress: (
                    self.timeline_slider.set(p)
                    if hasattr(self, "timeline_slider")
                    else None
                ),
            )
            self.after(
                0,
                lambda p=progress: (
                    self.timeline_label.configure(text=f"Posição: {p}%")
                    if hasattr(self, "timeline_label")
                    else None
                ),
            )
            partial_mag = 2 / N * abs(S)
            self.after(
                0, lambda n=n, ph=phasor, S=S: self.update_fourier_display(n, ph, S, N)
            )
            # Atualizar espectro parcial em tempo real
            self.after(
                0,
                lambda f=self.fourier_freq, pm=partial_mag, tm=target_mag: self.update_single_freq_spectrum(
                    f, pm, tm
                ),
            )
            n += 1
            time.sleep(self.fourier_step_ms / 1000.0)
        if n >= N:
            mag = 2 / N * abs(S)
            self.fourier_spectrum_data.append((self.fourier_freq, mag))
            self.after(0, self.finalize_fourier_animation)

    def run_fourier_scan(self):
        if len(getattr(self, "current_signal", [])) == 0:
            return
        freq_step = max(20, self.fs / len(self.current_signal))
        while self.fourier_scan_mode and self.fourier_freq <= self.fs / 2:
            N = len(self.current_signal)
            S = 0 + 0j
            for n in range(N):
                if not self.fourier_scan_mode:
                    break
                sample = self.current_signal[n]
                theta = 2 * np.pi * self.fourier_freq * n / self.fs
                S += sample * np.exp(-1j * theta)
                if n % 10 == 0:
                    progress = int(100 * self.fourier_freq / (self.fs / 2))
                    self.after(
                        0,
                        lambda p=progress: (
                            self.timeline_slider.set(p)
                            if hasattr(self, "timeline_slider")
                            else None
                        ),
                    )
            mag = 2 / N * abs(S)
            self.fourier_spectrum_data.append((self.fourier_freq, mag))
            self.after(0, self.update_spectrum_plot)
            self.fourier_freq += freq_step
            self.after(
                0,
                lambda f=self.fourier_freq: (
                    self.fourier_freq_slider.set(f)
                    if hasattr(self, "fourier_freq_slider")
                    else None
                ),
            )
            time.sleep(0.01)
        self.after(0, self.stop_fourier_animation)

    def update_fourier_display(self, n, phasor, S, N):
        try:
            if hasattr(self, "fourier_time_cursor"):
                self.fourier_time_cursor.set_xdata(
                    [self.current_time[n], self.current_time[n]]
                )
            else:
                self.fourier_time_cursor = self.ax_fourier_time.axvline(
                    self.current_time[n], color="yellow", alpha=0.6
                )
            self.ax_fourier_circle.clear()
            circle = np.exp(1j * np.linspace(0, 2 * np.pi, 100))
            self.ax_fourier_circle.plot(circle.real, circle.imag, "k--", alpha=0.3)
            self.ax_fourier_circle.plot(
                [0, phasor.real], [0, phasor.imag], "orange", lw=2
            )
            self.ax_fourier_circle.scatter(
                [phasor.real], [phasor.imag], c="orange", s=40
            )
            self.ax_fourier_circle.set_aspect("equal")
            self.setup_axis_style(
                self.ax_fourier_circle, "Real", "Imag", f"f={self.fourier_freq:.0f}Hz"
            )
            if not hasattr(self, "accum_path"):
                self.accum_path = deque(maxlen=200)
            self.accum_path.append((S.real, S.imag))
            self.ax_fourier_accum.clear()
            if len(self.accum_path) > 1:
                path = np.array(self.accum_path)
                self.ax_fourier_accum.plot(path[:, 0], path[:, 1], "cyan", alpha=0.7)
            self.ax_fourier_accum.scatter([S.real], [S.imag], c="cyan", s=40)
            self.setup_axis_style(
                self.ax_fourier_accum, "Real", "Imag", f"Soma: {abs(S):.3f}"
            )
            self.canvas_fourier.draw_idle()
        except Exception as e:
            print(f"Erro na atualização: {e}")

    def update_fourier_frame_at_position(self, position):
        if len(getattr(self, "current_signal", [])) == 0:
            return
        N = len(self.current_signal)
        n = int(N * position / 100)
        n = min(n, N - 1)
        S = 0 + 0j
        for i in range(n + 1):
            sample = self.current_signal[i]
            theta = 2 * np.pi * self.fourier_freq * i / self.fs
            S += sample * np.exp(-1j * theta)
        theta = 2 * np.pi * self.fourier_freq * n / self.fs
        phasor = np.exp(-1j * theta)
        self.update_fourier_display(n, phasor, S, N)

    def update_spectrum_plot(self):
        self.ax_fourier_spectrum.clear()
        if self.fourier_spectrum_data:
            freqs, mags = zip(*sorted(self.fourier_spectrum_data))
            self.ax_fourier_spectrum.plot(freqs, mags, "g-", lw=1.5)
        self.setup_axis_style(
            self.ax_fourier_spectrum, "Frequência (Hz)", "|X(f)|", "Espectro"
        )
        self.canvas_fourier.draw_idle()

    def update_single_freq_spectrum(self, freq, mag_partial, mag_target=None):
        """Mostra magnitude parcial acumulando durante a animação de uma única frequência.
        mag_target: magnitude esperada (FFT completa) para referência (linha tracejada).
        """
        if not hasattr(self, "ax_fourier_spectrum"):
            return
        self.ax_fourier_spectrum.clear()
        # Desenhar dados já varridos (se existirem) em segundo plano
        if self.fourier_spectrum_data:
            freqs, mags = zip(*sorted(self.fourier_spectrum_data))
            self.ax_fourier_spectrum.plot(
                freqs, mags, color="#16a34a", lw=1.2, alpha=0.6
            )
        # Ponto parcial atual
        self.ax_fourier_spectrum.scatter(
            [freq], [mag_partial], c="yellow", s=50, label="Parcial"
        )
        # Linha alvo
        if mag_target is not None:
            self.ax_fourier_spectrum.axhline(
                mag_target, color="#f87171", ls="--", lw=1, label="Alvo"
            )
            self.ax_fourier_spectrum.text(
                self.ax_fourier_spectrum.get_xlim()[0],
                mag_target,
                f" alvo≈{mag_target:.3f}",
                va="bottom",
                ha="left",
                fontsize=8,
                color="#f87171",
            )
        self.ax_fourier_spectrum.set_xlim(0, max(freq * 1.1, 10))
        current_max = max(
            [mag_partial]
            + ([max(mags)] if self.fourier_spectrum_data else [])
            + ([mag_target] if mag_target is not None else [])
        )
        self.ax_fourier_spectrum.set_ylim(
            0, current_max * 1.15 if current_max > 0 else 1
        )
        self.setup_axis_style(
            self.ax_fourier_spectrum, "Frequência (Hz)", "|X(f)|", "Espectro Parcial"
        )
        self.ax_fourier_spectrum.legend(loc="upper right", fontsize=8)
        self.canvas_fourier.draw_idle()

    def setup_fourier_animation(self):
        self.ax_fourier_time.clear()
        self.ax_fourier_time.plot(self.current_time, self.current_signal, "b-")
        self.setup_axis_style(self.ax_fourier_time, "Tempo (s)", "x[n]", "Sinal")
        self.accum_path = deque(maxlen=200)
        self.canvas_fourier.draw()

    def finalize_fourier_animation(self):
        self.update_spectrum_plot()
        self.fourier_anim_running = False
        if hasattr(self, "btn_fourier_play"):
            self.btn_fourier_play.configure(text="▶ Anima", fg_color="#10b981")


# ==================== Função Principal ====================


def main():
    """Executa a aplicação"""
    # Verificar dependências
    try:
        import importlib.util

        if importlib.util.find_spec("pyaudio") is None:
            raise ImportError
    except ImportError:
        print("Instalando PyAudio...")
        import subprocess

        subprocess.check_call(["pip", "install", "pyaudio"])

    app = ModernSignalSystemsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
