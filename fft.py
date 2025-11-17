import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from matplotlib import pyplot as plt
    return np, plt


@app.cell
def _():
    SAMPLE_RATE = 44100 #Hertz
    DURATION = 5 #Seconds
    return DURATION, SAMPLE_RATE


@app.cell
def _(np):
    def generate_sine_wave(freq, sample_rate, duration):
        x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
        frequencies = x * freq
        #2pi because np.sin takes radians
        y = np.sin((2 * np.pi) * frequencies)
        return x, y
    return (generate_sine_wave,)


@app.cell
def _(DURATION, SAMPLE_RATE, generate_sine_wave, plt):
    x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)
    plt.plot(x, y)
    plt.show()
    return


@app.cell
def _(DURATION, SAMPLE_RATE, generate_sine_wave):
    _, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
    _, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
    noise_tone = noise_tone * 0.3
    mixed_tone = nice_tone + noise_tone
    return (mixed_tone,)


@app.cell
def _(mixed_tone, np, plt):
    normalized_tone = np.int16((mixed_tone/mixed_tone.max()) * 32767)

    plt.plot(normalized_tone[:1000])
    plt.show()
    return (normalized_tone,)


@app.cell
def _(SAMPLE_RATE, normalized_tone):
    from scipy.io.wavfile import write

    write("mysinewave.wav", SAMPLE_RATE, normalized_tone)
    return (write,)


@app.cell
def _(DURATION, SAMPLE_RATE, normalized_tone, np, plt):
    from scipy.fft import fft, fftfreq

    # Number of samples in normalized_tone
    N = SAMPLE_RATE * DURATION

    yf = fft(normalized_tone)
    xf = fftfreq(N, 1 / SAMPLE_RATE)

    plt.plot(xf, np.abs(yf))
    plt.show()
    return (N,)


@app.cell
def _(N, SAMPLE_RATE, normalized_tone, np, plt):
    from scipy.fft import rfft, rfftfreq

    yrf = rfft(normalized_tone)
    xrf = rfftfreq(N, 1 / SAMPLE_RATE)

    plt.plot(xrf, np.abs(yrf))
    plt.show()
    return xrf, yrf


@app.cell
def _(SAMPLE_RATE, xrf):
    # The maximum frequency is half the sample rate
    points_per_freq = len(xrf) / (SAMPLE_RATE / 2)

    # Our target frequency is 4000 Hz
    target_idx = int(points_per_freq * 4000)

    return (target_idx,)


@app.cell
def _(np, plt, target_idx, xrf, yrf):
    yrf[target_idx - 1: target_idx + 2] = 0

    plt.plot(xrf, np.abs(yrf))
    plt.show()
    return


@app.cell
def _(plt, yrf):
    from scipy.fft import irfft

    new_sig = irfft(yrf)

    plt.plot(new_sig[:1000])
    plt.show()
    return (new_sig,)


@app.cell
def _(SAMPLE_RATE, new_sig, np, write):
    norm_new_sig = np.int16(new_sig * (32767 / new_sig.max()))

    write("clean.wav", SAMPLE_RATE, norm_new_sig)
    return


if __name__ == "__main__":
    app.run()
