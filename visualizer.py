import numpy as np
import math as mt
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
import matplotlib.style as mplstyle
from alive_progress import alive_bar

mplstyle.use("fast")

PLOTSIZE = 1920
FPS = 24
px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches

# samplerate, song_data = wavfile.read('sinus20hz.wav')
samplerate, song_data = wavfile.read("Vaatwasser_MetGitaar_final2.wav")

skipped_frames = samplerate // FPS
amount_of_frames = len(song_data) // skipped_frames
window_width = samplerate
# window_width = skipped_frames


def stereo_sound_generator(left_channel, right_channel, limit=0):
    samples_made = 0

    while left_channel.size > window_width and (limit == 0 or samples_made < limit):
        yield left_channel[0:window_width], right_channel[0:window_width]
        
        left_channel = left_channel[skipped_frames:-1]
        right_channel = right_channel[skipped_frames:-1]
        samples_made += 1


def progress_callback(i, n):
    time_per_frame = (time.time()-starting_time) / (i + 1)
    print(f"Saving frame {i+1}/{n} | time remaining: {time_per_frame * (n-i):.2f} s")


sound_gen = stereo_sound_generator(song_data[:, 0], song_data[:, 1])
sound_list = [i for i in sound_gen]

frame_nr = 0

logarithmic_range_1 = np.logspace(mt.log10(19), mt.log10(50), 8)
logarithmic_range_2 = np.logspace(mt.log10(logarithmic_range_1[-1]), mt.log10(3000), 30)
logarithmic_range_3 = np.logspace(
    mt.log10(logarithmic_range_2[-1]), mt.log10((window_width // 2)), 25
)

logarithmic_range = np.append(logarithmic_range_1, logarithmic_range_2[1:])
logarithmic_range = np.append(logarithmic_range, logarithmic_range_3[1:])

# frame = sound_gen.__next__()
frame = sound_list.pop(0)
# ideas 'borrowed' from https://github.com/aiXander/Realtime_PyAudio_FFT
links_fft = np.abs(np.fft.rfft(frame[0] * np.hamming(len(frame[0]))))
rechts_fft = np.abs(np.fft.rfft(frame[1] * np.hamming(len(frame[1]))))

links_scaled_histogram, _ = np.histogram(
    range(len(links_fft)),
    logarithmic_range,
    weights=links_fft,
)
rechts_scaled_histogram, _ = np.histogram(
    range(len(links_fft)),
    logarithmic_range,
    weights=rechts_fft,
)


def prepare_animation(bar_container):
    def animate(frame_data):
        links_fft = np.abs(np.fft.rfft(frame_data[0] * np.hamming(len(frame[0]))))
        rechts_fft = np.abs(np.fft.rfft(frame_data[1] * np.hamming(len(frame[1]))))
        # np.BUFSIZE

        links_scaled_histogram, _ = np.histogram(
            range(len(links_fft)),
            logarithmic_range,
            weights=links_fft,
        )
        rechts_scaled_histogram, _ = np.histogram(
            range(len(links_fft)),
            logarithmic_range,
            weights=rechts_fft,
        )
        y_bars = np.append(np.flip(rechts_scaled_histogram), links_scaled_histogram)

        for count, rect in zip(y_bars, bar_container.patches):
            rect.set_height(2 * count)
            rect.set_y(-count)
        return bar_container.patches

    return animate


plt.ioff()
fig, ax = plt.subplots(figsize=(PLOTSIZE * px, PLOTSIZE * px))
x_bars = np.append(
    range(len(rechts_scaled_histogram)), range(-len(links_scaled_histogram), 0)
)
y_bars = np.append(np.flip(rechts_scaled_histogram), links_scaled_histogram)
bar_container = ax.bar(x_bars, y_bars, width=1, align="edge", color="black")

ax.set_xlim(-60, 60)
ax.set_ylim(-2.5 * 1e13, 2.5 * 1e13)
ax.axis("off")

ani = animation.FuncAnimation(
    fig,
    prepare_animation(bar_container),
    sound_list,
    repeat=False,
    blit=True,
    interval=1000 / FPS,
)

starting_time = time.time()
fig.tight_layout()

with alive_bar(len(sound_list)) as bar:
    ani.save(
        "Vaatwasser.mp4",
        "ffmpeg",
        FPS,
        progress_callback=bar,
        savefig_kwargs={"pad_inches": 0},
    )

# while links_array.size > window_width:
#     links_fft = np.abs(np.fft.fft(links_array[0:window_width]))
#     rechts_fft = np.abs(np.fft.fft(rechts_array[0:window_width]))

#     links_scaled_histogram = np.histogram(range(len(links_fft)//2), logarithmic_range, weights=links_fft[:len(links_fft)//2])
#     rechts_scaled_histogram = np.histogram(range(len(links_fft)//2), logarithmic_range, weights=rechts_fft[:len(links_fft)//2])

#     plt.bar(range(-len(links_scaled_histogram[0]), 0), links_scaled_histogram[0], width=1, align='edge', color="black")
#     plt.bar(range(-len(links_scaled_histogram[0]), 0), -links_scaled_histogram[0], width=1, align='edge', color="black")
#     plt.bar(range(len(rechts_scaled_histogram[0])), np.flip(rechts_scaled_histogram[0]), width=1, align='edge', color="black")
#     plt.bar(range(len(rechts_scaled_histogram[0])), -np.flip(rechts_scaled_histogram[0]), width=1, align='edge', color="black")
#     plt.xlim(-60, 60)
#     plt.ylim(-2.5*1e13, 2.5*1e13)
#     plt.axis('off')
#     plt.savefig(f'K:/video/{frame_nr}.png', bbox_inches='tight', pad_inches = 0)
#     plt.clf()

#     links_array = links_array[skipped_frames:-1]
#     rechts_array = rechts_array[skipped_frames:-1]
#     frame_nr += 1
#     print(f"{frame_nr} / {amount_of_frames}")
