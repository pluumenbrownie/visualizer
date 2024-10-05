import numpy as np
from numpy.fft import rfft, rfftfreq
from numpy.typing import *
import math as mt
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
import matplotlib.style as mplstyle
from alive_progress import alive_bar
from typing import Any, Generator, Iterable, Self, Sequence
from pathlib import Path
from itertools import batched
# from manim import *


# config.frame_height = config.frame_width = 16
# config.pixel_height = config.pixel_width = 1920
# config.frame_rate = 24

mplstyle.use("fast")

PLOTSIZE = 1920
FPS = 24
PX = 1 / plt.rcParams["figure.dpi"]  # pixel in inches


class AudioStream:
    def __init__(self, audio_data: NDArray, window_size: int) -> None:
        self.data = audio_data
        self.window_size = window_size
        self.n = 0

    def __iter__(self):
        for segment in batched(self.data, self.window_size):
            yield np.array(segment, dtype=np.float64)


class Song:
    def __init__(self, path: Path, fps: float) -> None:
        self.samplerate, song_data = wavfile.read(path)
        window_size: int = round(self.samplerate / fps)
        self.left = AudioStream(song_data[:, 0], window_size)
        self.right = AudioStream(song_data[:, 1], window_size)
        self.frequency_distribution = LogarithmicRange(self.samplerate)

    def generate_histograms(self) -> list[Sequence[float]]:
        output = []
        bass_start = 19
        mid_start = 50
        high_start = 3000
        bass_bins = 8
        mid_bins = 30
        high_bins = 25
        
        # logarithmic_range_1 = np.logspace(mt.log10(bass_start), mt.log10(mid_start), bass_bins)
        # logarithmic_range_2 = np.logspace(mt.log10(logarithmic_range_1[-1]), mt.log10(high_start), mid_bins)
        # logarithmic_range_3 = np.logspace(
        #     mt.log10(logarithmic_range_2[-1]), mt.log10((self.samplerate // 2)), high_bins
        # )

        # logarithmic_range = np.append(logarithmic_range_1, logarithmic_range_2[1:])
        # logarithmic_range = np.append(logarithmic_range, logarithmic_range_3[1:])
        for left, right in zip(self.left, self.right):
            left_fft = np.abs(rfft(left))
            right_fft = np.abs(rfft(right))
            fft_freqs = rfftfreq(len(left), d=1/self.samplerate)

            left_histo, _ = np.histogram(
                fft_freqs,
                self.frequency_distribution.data,
                weights=left_fft,
            )
            right_histo, _ = np.histogram(
                fft_freqs,
                self.frequency_distribution.data,
                weights=right_fft,
            )
            merged_histo = np.append(np.flip(left_histo), right_histo)
            output.append(merged_histo)
        return output


# class Visualizer(Scene):
#     def construct(self):
#         song = Song(Path("tests/sinus.wav"), 24)
#         bar_data = song.generate_histograms()
#         bar_amount = len(bar_data[0])
#         print(f"{len(bar_data) = }")
#         bars = []
#         width_unit = 16*RIGHT/bar_amount
#         for n in range(bar_amount):
#             bar = Rectangle(height=1, width=width_unit[0], fill_opacity=1.0, stroke_color=np.array([0, 0, 0, 0]))
#             bar.set_stroke(opacity=0.0)
#             bar.shift(LEFT * 8 + (n + 0.5) * width_unit)
#             # bar.shift((n + 0.5) * width_unit)
#             bars.append(bar)
#             self.add(bar)
#         # bar_data = bar_data[0]
#         # print(f"{max(bar_data) = }")
#         scale_factor = 1e12
        
#         for frame_data in bar_data:
#             for bar, height in zip(bars, frame_data):
#                 bar.stretch_to_fit_height(1920 * height / scale_factor)
#             self.wait(1 / config.frame_rate)
        
#     def bar_animation(self, bars: list[Rectangle], bar_data: Sequence[float]) -> Iterable[Animation]:
#         output = []
#         scale_factor = 1e12
#         for bar, height in zip(bars, bar_data):
#             output.append(bar.stretch_to_fit_height(height / scale_factor))
#         return output
    

class LogarithmicRange:
    def __init__(self, samplerate: int) -> None:
        bass_start = 19
        mid_start = 50
        high_start = 3000
        bass_bins = 8
        mid_bins = 30
        high_bins = 25
        
        logarithmic_range_1 = np.logspace(mt.log10(bass_start), mt.log10(mid_start), bass_bins)
        logarithmic_range_2 = np.logspace(mt.log10(logarithmic_range_1[-1]), mt.log10(high_start), mid_bins)
        logarithmic_range_3 = np.logspace(
            mt.log10(logarithmic_range_2[-1]), mt.log10((samplerate // 2)), high_bins
        )

        logarithmic_range = np.append(logarithmic_range_1, logarithmic_range_2[1:])
        logarithmic_range = np.append(logarithmic_range, logarithmic_range_3[1:])
        self.data: NDArray[np.floating[Any]] = logarithmic_range
    
    def __iter__(self):
        for n in self.data:
            yield float(n)
    
    def __len__(self) -> int:
        return len(self.data)


class MatplotlibVisualizer:
    def __init__(self, song: Song, dimensions: tuple[int, int] = (1920, 1920), ) -> None:
        self.song = song
        self.x, self.y = dimensions

    def save_to(self, out_path: str):
        song_data = self.song.generate_histograms()
        def prepare_animation(bar_container):
            def animate(frame_data):

                for count, rect in zip(frame_data, bar_container.patches):
                    rect.set_height(2 * count)
                    rect.set_y(-count)
                return bar_container.patches
            return animate
        plt.ioff()
        fig, ax = plt.subplots(figsize=(self.x * PX, self.y * PX))
        x_bars = range(len(self.song.frequency_distribution))
        y_bars = np.zeros_like(x_bars)
        bar_container = ax.bar(x_bars, y_bars, width=1, align="edge", color="black")

        ax.set_xlim(0, len(x_bars))
        y_scale = 2.5 * 1e12
        ax.set_ylim(-y_scale, y_scale)
        ax.axis("off")

        ani = animation.FuncAnimation(
            fig,
            prepare_animation(bar_container),
            song_data,
            repeat=False,
            blit=True,
            interval=1000 / FPS,
        )

        starting_time = time.time()
        fig.tight_layout()

        with alive_bar(len(song_data)) as bar:
            def call_bar(amount: int, total: int):
                bar()

            ani.save(
                out_path,
                "ffmpeg",
                FPS,
                progress_callback=call_bar,
                savefig_kwargs={"pad_inches": 0},
            )


in_path = Path("input/Vaatwasser_MetGitaar_final2.wav")
song = Song(in_path, 24.0)
visualizer = MatplotlibVisualizer(song)
visualizer.save_to("out/vaatwasser.mp4")


# PLOTSIZE = 1920
# FPS = 24
# px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches

# samplerate, song_data = wavfile.read('tests/sinus.wav')
# # samplerate, song_data = wavfile.read("input/Vaatwasser_MetGitaar_final2.wav")
# out_name = "sinus.mp4"

# # seg_len = 2 * samplerate
# # print(f"{samplerate = }")
# # song_data = np.sin(2*np.pi*(np.linspace(0, seg_len, seg_len))/440 )

# skipped_frames = samplerate // FPS
# amount_of_frames = len(song_data) // skipped_frames
# window_width = samplerate
# # window_width = skipped_frames

# song_fft = np.abs(rfft(song_data[:, 0]))
# fft_freqs = rfftfreq(len(song_data[:, 0]), d=1/samplerate)
# # song_fft = rfft(song_data)
# # fft_freqs = rfftfreq(len(song_data), d=1/samplerate)

# plt.plot(fft_freqs, song_fft)
# plt.xlim(0, 1000)
# plt.show()