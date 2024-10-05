import numpy as np
import math as mt
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
from numpy.fft import rfft, rfftfreq
import matplotlib.style as mplstyle
from alive_progress import alive_bar
from typing import Iterable, Self, Sequence
from pathlib import Path
from itertools import batched
from manim import *


config.frame_height = config.frame_width = 16
config.pixel_height = config.pixel_width = 1920
config.frame_rate = 24

mplstyle.use("fast")

PLOTSIZE = 1920
FPS = 24
px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches


class AudioStream:
    def __init__(self, audio_data: Sequence[float], window_size: int) -> None:
        self.data = audio_data
        self.window_size = window_size
        self.n = 0

    def __iter__(self):
        for segment in batched(self.data, self.window_size):
            yield np.array(segment, dtype=np.float64)


class Song:
    def __init__(self, path: Path, fps: float) -> None:
        self.samplerate, song_data = wavfile.read(path)
        window_size = self.samplerate // fps
        self.left = AudioStream(song_data[:, 0], window_size)
        self.right = AudioStream(song_data[:, 1], window_size)

    def generate_histograms(self) -> list[Sequence[float]]:
        output = []
        bass_start = 19
        mid_start = 50
        high_start = 3000
        bass_bins = 8
        mid_bins = 30
        high_bins = 25
        
        logarithmic_range_1 = np.logspace(mt.log10(bass_start), mt.log10(mid_start), bass_bins)
        logarithmic_range_2 = np.logspace(mt.log10(logarithmic_range_1[-1]), mt.log10(high_start), mid_bins)
        logarithmic_range_3 = np.logspace(
            mt.log10(logarithmic_range_2[-1]), mt.log10((self.samplerate // 2)), high_bins
        )

        logarithmic_range = np.append(logarithmic_range_1, logarithmic_range_2[1:])
        logarithmic_range = np.append(logarithmic_range, logarithmic_range_3[1:])
        for left, right in zip(self.left, self.right):
            left_fft = np.abs(rfft(left))
            right_fft = np.abs(rfft(right))
            fft_freqs = rfftfreq(len(left), d=1/self.samplerate)

            left_histo, _ = np.histogram(
                fft_freqs,
                logarithmic_range,
                weights=left_fft,
            )
            right_histo, _ = np.histogram(
                fft_freqs,
                logarithmic_range,
                weights=right_fft,
            )
            merged_histo = np.append(np.flip(left_histo), right_histo)
            output.append(merged_histo)
        return output


class Visualizer(Scene):
    def construct(self):
        song = Song(Path("tests/sinus.wav"), 24)
        bar_data = song.generate_histograms()
        bar_amount = len(bar_data[0])
        print(f"{len(bar_data) = }")
        bars = []
        width_unit = 16*RIGHT/bar_amount
        for n in range(bar_amount):
            bar = Rectangle(height=1, width=width_unit[0], fill_opacity=1.0, stroke_color=np.array([0, 0, 0, 0]))
            bar.set_stroke(opacity=0.0)
            bar.shift(LEFT * 8 + (n + 0.5) * width_unit)
            # bar.shift((n + 0.5) * width_unit)
            bars.append(bar)
            self.add(bar)
        # bar_data = bar_data[0]
        # print(f"{max(bar_data) = }")
        scale_factor = 1e12
        
        for frame_data in bar_data:
            for bar, height in zip(bars, frame_data):
                bar.stretch_to_fit_height(1920 * height / scale_factor)
            self.wait(1 / config.frame_rate)
        
    def bar_animation(self, bars: list[Rectangle], bar_data: Sequence[float]) -> Iterable[Animation]:
        output = []
        scale_factor = 1e12
        for bar, height in zip(bars, bar_data):
            output.append(bar.stretch_to_fit_height(height / scale_factor))
        return output
    

class MatplotlibVisualizer:
    def __init__(self, song: Song) -> None:
        self.song = song

    def save_to(self, out_path: str):
        song_data = self.song.generate_histograms()




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