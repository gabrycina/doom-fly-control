"""Soundtrack for demo v2. All music synthesised here (numpy/scipy). Game SFX: Freedoom 0.13.0 lumps (BSD-3-Clause)
from the freedoom2.wad shipped with vizdoom. Neuron crackle: density follows the REAL simulated spike counts."""
import json, subprocess
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve, resample_poly
ROOT = Path(__file__).resolve().parents[1]
import imageio_ffmpeg; FF = imageio_ffmpeg.get_ffmpeg_exe()
SR = 48000; DUR = 57.8; N = int(DUR * SR); BEAT = 0.6
rng = np.random.default_rng(11)
mus = np.zeros((N, 2)); sfx = np.zeros((N, 2))

def sos(kind, f, order=2): return butter(order, f, btype=kind, fs=SR, output="sos")
def lp(x, f): return sosfilt(sos("low", f), x)
def hp(x, f): return sosfilt(sos("high", f), x)
def bp(x, lo, hi): return sosfilt(sos("band", [lo, hi]), x)
def env(n, a=0.002, d=0.2):
    t = np.arange(n) / SR; return np.minimum(t / max(a, 1e-6), 1) * np.exp(-t / d)
def add(buf, t, x, gain=1.0, pan=0.0):
    i = int(t * SR)
    if i >= N or i + len(x) <= 0: return
    if i < 0: x = x[-i:]; i = 0
    x = x[:N - i]; l, r = np.sqrt((1 - pan) / 2), np.sqrt((1 + pan) / 2)
    buf[i:i + len(x), 0] += x * gain * l * 1.414; buf[i:i + len(x), 1] += x * gain * r * 1.414
def saw(f, n, det=0.0):
    t = np.arange(n) / SR; return 2 * ((t * f * (1 + det)) % 1) - 1
def note(m): return 440 * 2 ** ((m - 69) / 12)

# ---------------------------------------------------------------- instruments
def kick(g=1.0):
    n = int(0.4 * SR); t = np.arange(n) / SR; f = 45 + 90 * np.exp(-t / 0.03)
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18) + 0.3 * np.exp(-t / 0.003) * rng.standard_normal(n) * 0.3) * g
def clap():
    n = int(0.25 * SR); x = bp(rng.standard_normal(n), 900, 3500)
    e = sum(env(n, 0.0005, 0.012) * (np.arange(n) >= int(k * SR)) for k in (0, 0.01, 0.02)) + env(n, 0.001, 0.08) * 0.6
    return x * e * 0.8
def hat(d=0.03): n = int(0.12 * SR); return hp(rng.standard_normal(n), 7000) * env(n, 0.0005, d) * 0.5
def bass(m, dur, cut=400):
    n = int(dur * SR); x = saw(note(m), n) + saw(note(m), n, 0.004); return lp(x, cut) * env(n, 0.005, dur * 0.7) * 0.6
def pad(ms, dur, cut=1200, att=0.6):
    n = int(dur * SR); x = sum(saw(note(m), n, d) for m in ms for d in (-0.003, 0.003)) / (2 * len(ms))
    t = np.arange(n) / SR; e = np.minimum(t / att, 1) * np.minimum((dur - t) / 0.5, 1).clip(0, 1)
    return lp(x, cut) * e
def pluck(m, dur=0.35):
    n = int(dur * SR); t = np.arange(n) / SR; f = note(m)
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.exp(-t / 0.12) * 0.4
def stab(ms, dur=0.35):
    n = int(dur * SR); x = sum(saw(note(m), n) for m in ms) / len(ms); return lp(x, 2500) * env(n, 0.002, 0.12)
def riser(dur, f0=300, f1=6000):
    n = int(dur * SR); t = np.arange(n) / SR; x = rng.standard_normal(n); out = np.zeros(n); blk = 2400
    for s in range(0, n, blk):
        fc = f0 * (f1 / f0) ** (s / n); out[s:s + blk] = bp(x[s:s + blk + 0], fc * 0.7, min(fc * 1.3, 20000))[:len(out[s:s + blk])]
    tone = np.sin(2 * np.pi * np.cumsum(110 * 4 ** (t / dur)) / SR) * 0.3
    return (out + tone) * (t / dur) ** 2
def hit(g=1.0):
    n = int(2.2 * SR); t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(38 + 60 * np.exp(-t / 0.05)) / SR) * np.exp(-t / 0.6)
    crash = hp(rng.standard_normal(n), 3000) * np.exp(-t / 0.7) * 0.35
    return (boom + crash) * g
def drone(dur, m=33):
    n = int(dur * SR); t = np.arange(n) / SR
    x = lp(saw(note(m), n) + saw(note(m), n, 0.006) + 0.5 * saw(note(m + 7), n, -0.004), 380) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.2 * t))
    return x * 0.35
def reverb(x, secs=1.8, wet=0.25):
    n = int(secs * SR); ir = rng.standard_normal((n, 2)) * np.exp(-np.arange(n) / SR / (secs / 5))[:, None]; ir /= np.abs(ir).sum(0) ** 0.5 * 8
    y = np.stack([fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1); return x + wet * y

CH = [[57, 60, 64], [53, 57, 60], [50, 53, 57], [52, 56, 59]]      # Am F Dm E
ROOTS = [33, 29, 26, 28]
def bar_of(t): return int(t // 2.4) % 4

# ---------------------------------------------------------------- music sections
add(mus, 0, drone(3.6), 0.5)
for k in range(6): add(mus, 0.4 + k * BEAT, kick(0.35) * 0.6)                 # hook pulse on the loop-seam grid
for k in range(12): add(mus, 0.4 + k * 0.3, hat(0.02), 0.25, 0.3 if k % 2 else -0.3)
# context: airy pad + arpeggio
for b in range(3):
    t0 = 3.4 + b * 2.4; add(mus, t0, pad([m + 12 for m in CH[b % 4]] + [CH[b % 4][0] + 26], 2.6, 1800, 0.8), 0.6)
    for k in range(8): add(mus, t0 + k * 0.3, pluck(CH[b % 4][k % 3] + 24), 0.35, (-0.5, 0.5)[k % 2])
# question: dark pad, heartbeat, riser
add(mus, 10.6, pad([45, 48, 52], 4.6, 700, 0.3), 0.7); add(mus, 10.6, drone(4.4, 33), 0.4)
for t in np.arange(12.6, 15.0, BEAT): add(mus, t, kick(0.5))
add(mus, 13.4, riser(1.6), 0.35)
# version 1 groove (15.0 - 27.4), then drop at 27.4, groove to 29.6
for t in np.arange(15.0, 29.6, BEAT):
    b = bar_of(t); prog = (t - 15.0) / 12.4
    if not (26.9 < t < 27.4): add(mus, t, kick(0.9))
    for h in (0, 0.3): add(mus, t + h, hat(0.025), 0.28 if t < 24.4 else 0.4, 0.2)
    if t >= 24.4 or t >= 27.4:
        add(mus, t + 0.15, hat(0.015), 0.2, -0.2); add(mus, t + 0.45, hat(0.015), 0.2, -0.2)
    if t >= 20.0 and int(round((t - 15.0) / BEAT)) % 2 == 1: add(mus, t, clap(), 0.5)
    cut = 250 + 1300 * min(prog, 1) if t < 27.4 else 1600
    for h in (0, 0.3): add(mus, t + h, bass(ROOTS[b] + (12 if h else 0), 0.28, cut), 0.7)
for t0 in np.arange(15.0, 29.6, 2.4): add(mus, t0, pad(CH[bar_of(t0)], 2.4, 900), 0.35)
add(mus, 26.0, riser(1.4), 0.4); add(mus, 27.4, hit(1.0), 0.9)
# second fly: break + tension
add(mus, 29.6, drone(2.8, 31), 0.5); add(mus, 29.6, pad([43, 46, 50], 2.9, 600, 0.2), 0.6)
for k, t in enumerate(np.arange(29.6, 32.4, 0.15)): add(mus, t, bass(31 + (k // 8), 0.12, 300 + 60 * k), 0.4)
add(mus, 31.2, riser(1.2), 0.3)
for t in np.arange(32.4, 37.4, BEAT):                                          # real brain: sparse half-time
    k = int(round((t - 32.4) / BEAT))
    if k % 2 == 0: add(mus, t, kick(0.7))
    add(mus, t, hat(0.02), 0.15)
add(mus, 32.4, pad([45, 48, 52, 55], 5.1, 700, 0.8), 0.45)
add(mus, 37.4, pad([45, 51, 56, 58], 3.9, 900, 0.1), 0.5)                     # scrambled: dissonant
for t in np.arange(37.4, 41.2, 0.15): add(mus, t, hat(0.01), 0.3 * rng.random(), rng.uniform(-0.8, 0.8))
for t in np.arange(37.4, 41.2, BEAT): add(mus, t, kick(0.6))
add(mus, 41.2, drone(1.9, 30), 0.5); add(mus, 41.3, riser(1.8, 200, 8000), 0.45)
# kinetic section: stabs on the words (dotted rhythm on the 100 BPM grid), kick on every beat
for t in np.arange(43.2, 53.2, BEAT):
    add(mus, t, kick(1.0)); add(mus, t, hat(0.02), 0.3); add(mus, t + 0.3, hat(0.03), 0.35)
    if int(round((t - 43.2) / BEAT)) % 2 == 1: add(mus, t, clap(), 0.45)
    for h in (0, 0.3): add(mus, t + h, bass(ROOTS[bar_of(t)] + (12 if h else 0), 0.28, 1400), 0.6)
for tw in (43.2, 44.1, 45.0, 45.9): add(mus, tw, stab([57, 60, 64, 69]), 0.8)
for t0 in np.arange(46.8, 53.2, 2.4): add(mus, t0, pad(CH[bar_of(t0)], 2.4, 1100), 0.3)
# final hit + tail that settles into the hook's drone/pulse for the loop
add(mus, 53.2, hit(1.1), 1.0); add(mus, 53.2, pad([45, 48, 52, 57], 4.6, 900, 0.05), 0.5)
add(mus, 54.4, drone(3.4), 0.5)
for t in (56.2, 56.8, 57.4): add(mus, t + 0.2, kick(0.35) * 0.6)               # 56.4, 57.0, 57.6 -> seam -> 0.4
mus = reverb(mus, 1.8, 0.2)

# ---------------------------------------------------------------- SFX
def whoosh(dur=0.45):
    n = int(dur * SR); t = np.arange(n) / SR; x = rng.standard_normal(n); out = np.zeros(n)
    for s in range(0, n, 1200):
        fc = 400 * (8 ** np.sin(np.pi * s / n)); out[s:s + 1200] = bp(x[s:s + 1200], fc * 0.6, fc * 1.4)
    return out * np.sin(np.pi * t / dur) ** 2 * 0.6
def pop():
    n = int(0.08 * SR); t = np.arange(n) / SR; return np.sin(2 * np.pi * np.cumsum(900 * np.exp(-t / 0.02) + 300) / SR) * np.exp(-t / 0.025) * 0.5
def tick(): n = int(0.02 * SR); t = np.arange(n) / SR; return np.sin(2 * np.pi * 2400 * t) * np.exp(-t / 0.004) * 0.4
def glitch(dur=0.35):
    n = int(dur * SR); x = np.zeros(n); s = 0
    while s < n:
        L = int(rng.uniform(0.01, 0.05) * SR); f = rng.uniform(80, 2000)
        seg = np.sign(np.sin(2 * np.pi * f * np.arange(L) / SR)) * 0.4 if rng.random() < 0.5 else rng.standard_normal(L) * 0.5
        x[s:s + L] = seg[:n - s] * (rng.random() > 0.2); s += L
    return hp(x, 150) * 0.7
def beep(f=1400, d=0.09):
    n = int(d * SR); t = np.arange(n) / SR; return np.sign(np.sin(2 * np.pi * f * t)) * 0.25 * np.minimum(1, (d - t) / 0.01)
def servo(d=0.5):
    n = int(d * SR); t = np.arange(n) / SR; f = 220 + 260 * np.sin(np.pi * t / d)
    return lp(np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)), 1500) * 0.25 * np.sin(np.pi * t / d)
def thud(): n = int(0.6 * SR); t = np.arange(n) / SR; return np.sin(2 * np.pi * np.cumsum(30 + 50 * np.exp(-t / 0.08)) / SR) * np.exp(-t / 0.25)

for t in (3.4, 10.6, 15.0, 20.0, 24.4, 29.6, 32.4, 41.2, 46.8, 53.2): add(sfx, t - 0.2, whoosh(), 0.5, rng.uniform(-0.3, 0.3))
add(sfx, 8.95, whoosh(0.3), 0.6)
CAPTIONS = [0.0, 0.5, 3.5, 4.0, 8.0, 9.0, 10.6, 10.95, 12.8, 13.2, 15.0, 17.2, 20.0, 22.4, 24.4, 27.9,
            29.6, 30.3, 30.8, 32.4, 37.4, 39.0, 41.2, 42.0, 46.8, 47.8, 53.6]
for t in CAPTIONS: add(sfx, t, pop(), 0.35)
for t in (15.4, 15.85, 16.3, 20.3, 21.3): add(sfx, t, pop(), 0.25, 0.4)       # boxes landing
for t0, t1 in ((5.6, 7.4), (24.6, 26.1), (49.2, 50.4)):                       # count-ups
    t = t0
    while t < t1: add(sfx, t, tick(), 0.35, rng.uniform(-0.4, 0.4)); t += 0.04 + 0.12 * ((t - t0) / (t1 - t0)) ** 2
for t, d in ((0.0, 0.4), (13.0, 1.0), (20.8, 0.4), (37.4, 0.6)): add(sfx, t, glitch(d), 0.55)
add(sfx, 41.1, thud(), 0.6)
add(sfx, 46.8, beep(1400), 0.6, 0.5); add(sfx, 46.95, beep(1800), 0.6, 0.5); add(sfx, 47.1, servo(), 0.8, 0.5)
add(sfx, 47.8, beep(1200, 0.12), 0.5, 0.5); add(sfx, 47.95, servo(0.35), 0.6, 0.5)

# Freedoom game sounds (BSD-3-Clause), at real muzzle-flash / kill events in the footage shown
fd = np.load(ROOT / "footage/v2/freedoom_sfx.npz")
def fdsnd(name):
    x = (fd[name].astype(np.float32) - 128) / 128; r = int(fd[name + "_rate"])
    return resample_poly(x, SR, r)
PISTOL, DEATH = fdsnd("DSPISTOL"), fdsnd("DSPODTH1")
def flashes(frames):
    r = frames[:, 120:200, 120:200].astype(np.int16)
    m = ((r[..., 0] > 220) & (r[..., 1] > 150) & (r[..., 2] < 140)).reshape(len(frames), -1).sum(1) > 30
    return np.flatnonzero(np.diff(m.astype(int)) == 1) + 1
def events(frames, kills, t0, t1, tic_of, gain, pan, max_rate=None):
    fl = set(flashes(frames).tolist()); last = -1; lastshot = -9
    for i in range(int((t1 - t0) * SR / 480)):
        t = t0 + i * 480 / SR; tic = min(tic_of(t - t0), len(frames) - 1)
        for k in range(last + 1, tic + 1):
            if k in fl and t - lastshot > (max_rate or 0): add(sfx, t, PISTOL, gain, pan); lastshot = t
            if k > 0 and kills[k] > kills[k - 1]: add(sfx, t, DEATH, gain * 0.8, pan)
        last = max(last, tic)
DTC = {a: np.load(ROOT / f"footage/{a}_dtc_ep0.npz") for a in ["real", "shuf0", "noconn"]}
for a, pan in (("real", -0.4), ("shuf0", 0.0), ("noconn", 0.4)):
    events(DTC[a]["frames"], DTC[a]["kills"], 0.0, 3.4, lambda u: int((0.8 + u) * 35 * 2.5), 0.05, pan, 0.1)
SPR = np.load(ROOT / "footage/v2/real_ep0_spikes.npz"); SPS = np.load(ROOT / "footage/v2/shuf0_ep0_spikes.npz")
ROB = np.load(ROOT / "footage/a2/spray_matched_ep0.npz")
events(SPR["frames"], SPR["kills"], 32.4, 37.4, lambda u: int(u * 35), 0.14, -0.4)
events(SPS["frames"], SPS["kills"], 37.4, 41.2, lambda u: int(u * 35 * 1.6), 0.14, -0.4)
events(SPR["frames"], SPR["kills"], 43.2, 46.8, lambda u: int((u + 6) * 35 * 2), 0.10, 0.0, 0.12)
events(SPR["frames"], SPR["kills"], 46.8, 53.2, lambda u: int((u + 1) * 35 * 3), 0.08, -0.5, 0.12)
events(ROB["frames"], ROB["kills"], 46.8, 53.2, lambda u: int((u + 1) * 35 * 3), 0.08, 0.5, 0.12)

# neuron crackle: click density = real simulated spike count per game tic (all neurons), same tic mapping as the picture
def crackle(sp, t0, t1, tic_of, gain, per=600.0):
    ptr = sp["sp_ptr"]; cnt = np.add.reduceat(sp["sp_cnt"].astype(np.int64), ptr[:-1]) if len(ptr) > 1 else np.zeros(1)
    cnt[np.diff(ptr) == 0] = 0
    n = int((t1 - t0) * SR); x = np.zeros((n, 2)); blk = 480
    for s in range(0, n, blk):
        u = s / SR; tic = min(tic_of(u), len(cnt) - 1); spd = (tic_of(u + 0.1) - tic_of(u)) / 0.1 if u + 0.1 < t1 - t0 else 35
        rate = cnt[max(tic, 0)] * max(spd, 1) / per
        k = rng.poisson(rate * blk / SR)
        pos = rng.integers(0, blk, k) + s; pos = pos[pos < n]
        amp = rng.uniform(0.2, 1.0, len(pos)); pn = rng.uniform(0, 1, len(pos))
        np.add.at(x[:, 0], pos, amp * np.sqrt(1 - pn)); np.add.at(x[:, 1], pos, amp * np.sqrt(pn))
    x = np.stack([bp(x[:, c], 1500, 9000) for c in range(2)], 1)
    fade = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.15 * SR))[:, None]
    i = int(t0 * SR); sfx[i:i + n] += x * fade * gain
    return float(cnt.mean())
m_real_ctx = crackle(SPR, 6.6, 9.0, lambda u: int(u * 35), 0.08)
m_real = crackle(SPR, 32.4, 37.4, lambda u: int(u * 35), 0.08)
m_shuf = crackle(SPS, 37.4, 41.2, lambda u: int(u * 35 * 1.6), 0.08)
print("mean spikes/tic real", m_real, "scrambled", m_shuf)

# ---------------------------------------------------------------- mix + loudness
mix = mus * 0.55 + sfx * 1.0
edge = int(0.02 * SR); mix[:edge] *= np.linspace(0.6, 1, edge)[:, None]; mix[-edge:] *= np.linspace(1, 0.6, edge)[:, None]
mix /= np.abs(mix).max() * 1.05
wav = ROOT / "footage/v2/mix.wav"
import wave
with wave.open(str(wav), "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())
meas = subprocess.run([FF, "-hide_banner", "-i", str(wav), "-af", "loudnorm=I=-14:TP=-3.2:LRA=11:print_format=json", "-f", "null", "-"],
                      capture_output=True, text=True).stderr
j = json.loads(meas[meas.rindex("{"):meas.rindex("}") + 1])
af = (f"loudnorm=I=-14:TP=-3.2:LRA=11:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
      f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true,aresample=48000")
out = ROOT / "demo/demo.mp4"
subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(ROOT / "demo/demo_silent.mp4"), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-af", af, "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest", "-movflags", "+faststart", str(out)], check=True)
chk = subprocess.run([FF, "-hide_banner", "-i", str(out), "-map", "0:a", "-af", "loudnorm=I=-14:TP=-3.2:print_format=json", "-f", "null", "-"],
                     capture_output=True, text=True).stderr
k = json.loads(chk[chk.rindex("{"):chk.rindex("}") + 1])
print("final integrated LUFS", k["input_i"], "true peak", k["input_tp"], "size MB", out.stat().st_size / 1e6)
if __name__ == "__main__":
    for a, b, lab in [(0, 3.4, "hook"), (3.4, 10.6, "context"), (10.6, 15, "question"), (15, 27.4, "v1"), (27.4, 29.6, "drop"), (29.6, 32.4, "open"),
                      (32.4, 37.4, "real brain"), (37.4, 41.2, "scrambled"), (41.2, 43.2, "tension"), (43.2, 46.8, "kinetic"), (46.8, 53.2, "robot"), (53.2, 57.8, "end")]:
        s = slice(int(a * SR), int(b * SR)); r = lambda x: 20 * np.log10(np.sqrt((x[s] ** 2).mean()) + 1e-9)
        print(f"{lab:11s} music {r(mus * 0.55):6.1f} dB  sfx {r(sfx):6.1f} dB")
