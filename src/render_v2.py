"""v2 story cut: 1080x1080, 30 fps. Every number is read from results/ files; brain dots are real MaleCNS
soma positions; lit dots are real simulated spikes from the nftechie/doomfly run (record_spikes.py)."""
import json, math, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
import imageio_ffmpeg; FF = imageio_ffmpeg.get_ffmpeg_exe()
S, FPS = 1080, 30
BG = (14, 14, 13); INK = (255, 255, 255); INK2 = (200, 199, 190); MUTED = (130, 129, 122)
BLUE = (57, 135, 229); ORANGE = (217, 89, 38); GREEN = (25, 170, 115)
FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"; FR = "/System/Library/Fonts/Supplemental/Arial.ttf"
_fc = {}
def F(sz, bold=True):
    k = (sz, bold)
    if k not in _fc: _fc[k] = ImageFont.truetype(FB if bold else FR, sz)
    return _fc[k]
def ease(x): x = min(max(x, 0.0), 1.0); return 1 - (1 - x) ** 3
def clamp01(x): return min(max(x, 0.0), 1.0)

# ---------------------------------------------------------------- numbers (from results, never typed)
a1 = json.loads((ROOT / "results/a1/summary.json").read_text())
KILLS1 = {a: a1[f"defend_the_center/{a}"]["kills_mean"] for a in ["real", "shuf0", "noconn"]}
a2 = json.loads((ROOT / "results/a2/summary.json").read_text())
REAL2, SHUF2, ROBOT2 = a2["real"], a2["shuffled (3 seeds pooled)"], a2["spray_matched"]
N_NEURONS = 166700

# ---------------------------------------------------------------- footage
def load(p): return np.load(ROOT / p)
DTC = {a: load(f"footage/{a}_dtc_ep0.npz") for a in ["real", "shuf0", "noconn"]}
SP_REAL = load("footage/v2/real_ep0_spikes.npz"); SP_SHUF = load("footage/v2/shuf0_ep0_spikes.npz")
ROBOT = load("footage/a2/spray_matched_ep0.npz")
REAL_FR, REAL_K, REAL_CTRL = SP_REAL["frames"], SP_REAL["kills"], SP_REAL["ctrl"]
ROB_FR, ROB_K = ROBOT["frames"], ROBOT["kills"]

# ---------------------------------------------------------------- brain point cloud (real soma positions)
P = np.load(ROOT / "footage/v2/soma_pos.npz")["pos"]
HAS = ~np.isnan(P[:, 0])
PI = np.flatnonzero(HAS)                      # neuron indices with a soma position
Q = P[HAS].astype(np.float32)
Q = Q - (Q.min(0) + Q.max(0)) / 2
Q /= (Q[:, 2].max() - Q[:, 2].min())           # z (body axis) spans 1.0
POSROW = -np.ones(len(P), np.int64); POSROW[PI] = np.arange(len(PI))
rng = np.random.default_rng(7)
ORDER = rng.permutation(len(Q))                # reveal order for the count-up
SCRAMBLE_TGT = Q[rng.permutation(len(Q))]      # stylised "scramble" targets (illustration only)

def brain(w, h, theta, act=None, color=BLUE, n_show=None, base=0.55, zoom=1.0, morph=0.0, cz=0.0):
    """Render the CNS point cloud rotated by theta about the body axis. act: per-point activity (len Q)."""
    q = Q if morph <= 0 else Q * (1 - morph) + SCRAMBLE_TGT * morph
    c, s = math.cos(theta), math.sin(theta)
    X = q[:, 0] * c - q[:, 1] * s; Y = q[:, 2]; D = q[:, 0] * s + q[:, 1] * c
    sc = 0.92 * h * zoom
    xi = (w / 2 + X * sc).astype(np.int32); yi = (h / 2 + (Y - cz) * sc).astype(np.int32)
    m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if n_show is not None:
        vis = np.zeros(len(Q), bool); vis[ORDER[:n_show]] = True; m &= vis
    dens = np.zeros((h, w), np.float32)
    np.add.at(dens, (yi[m], xi[m]), (0.75 + 0.25 * (D[m] - D.min()) / (np.ptp(D) + 1e-9)))
    dens = np.clip(np.log1p(dens) / np.log1p(3.0), 0, 1.3) * base
    img = np.zeros((h, w, 3), np.float32) + np.array(BG, np.float32)
    img += dens[..., None] * np.array([150, 165, 195], np.float32)[None, None]
    if act is not None:
        g = np.zeros((h, w), np.float32)
        mm = m & (act > 0)
        np.add.at(g, (yi[mm], xi[mm]), act[mm])
        gi = Image.fromarray((np.clip(g * 120, 0, 255)).astype(np.uint8))
        core = np.asarray(gi.filter(ImageFilter.MaxFilter(3)), np.float32) / 255
        glow = np.asarray(gi.filter(ImageFilter.GaussianBlur(6)), np.float32) / 255
        col = np.array(color, np.float32)
        img += np.clip(glow * 5, 0, 1)[..., None] * col * 1.3 + core[..., None] * (0.5 * col + 0.6 * np.array([255, 255, 255], np.float32))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

class Activity:
    """Decaying per-point activity from recorded per-tic spikes (only neurons with a mapped soma are drawn)."""
    def __init__(self, sp):
        self.idx, self.ptr = sp["sp_idx"], sp["sp_ptr"]; self.a = np.zeros(len(Q), np.float32); self.t = 0
    def advance_to(self, tic):
        tic = min(tic, len(self.ptr) - 1)
        self.a *= 0.55
        while self.t < tic:
            nz = self.idx[self.ptr[self.t]:self.ptr[self.t + 1]]
            r = POSROW[nz]; r = r[r >= 0]
            self.a[r] += 1.0; self.t += 1
        return self.a

# ---------------------------------------------------------------- drawing helpers
def canvas(): return Image.new("RGB", (S, S), BG)

def wrap(d, s, font, maxw):
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= maxw: cur = t
        else: lines.append(cur); cur = w
    return lines + [cur]

def caption(im, text, y, t, size=66, color=INK, maxw=980, per_word=0.07, shadow=True):
    """Centered caption, words pop in one by one starting at local time t=0."""
    d = ImageDraw.Draw(im); font = F(size)
    lines = wrap(d, text, font, maxw); k = 0
    for li, line in enumerate(lines):
        words = line.split(); lw = d.textlength(line, font=font); x = S / 2 - lw / 2
        yy = y + li * int(size * 1.18)
        for w in words:
            p = clamp01((t - k * per_word) / 0.12); k += 1
            if p <= 0: x += d.textlength(w + " ", font=font); continue
            sz = int(size * (0.7 + 0.3 * ease(p)))
            f2 = F(sz)
            cx = x + d.textlength(w, font=font) / 2
            col = tuple(int(c * p + b * (1 - p)) for c, b in zip(color, BG))
            if shadow: d.text((cx + 3, yy + size / 2 + 3), w, font=f2, fill=(0, 0, 0), anchor="mm")
            d.text((cx, yy + size / 2), w, font=f2, fill=col, anchor="mm")
            x += d.textlength(w + " ", font=font)
    return y + len(lines) * int(size * 1.18)

def small(im, text, xy, size=24, color=MUTED, anchor="mm", bold=False):
    ImageDraw.Draw(im).text(xy, text, font=F(size, bold), fill=color, anchor=anchor)

def panel(fr, w, h): return Image.fromarray(fr).resize((w, h), Image.BILINEAR)

def progress(im, frac):
    d = ImageDraw.Draw(im); d.rectangle([0, S - 6, S, S], fill=(35, 35, 33)); d.rectangle([0, S - 6, int(S * frac), S], fill=(90, 90, 86))

# ---------------------------------------------------------------- stylised fly (own drawing, illustration)
def draw_fly(scale=1.0, cx=540, cy=560, wing_phase=0.0):
    W = S * 2; im = Image.new("RGBA", (W, W), BG + (255,)); d = ImageDraw.Draw(im)
    def T(x, y): return (cx * 2 + x * 2 * scale, cy * 2 + y * 2 * scale)
    def ell(x0, y0, x1, y1, **kw):
        a, b = T(x0, y0), T(x1, y1); d.ellipse([a[0], a[1], b[0], b[1]], **kw)
    legc = (70, 55, 40)
    for side in (-1, 1):
        for (ax, ay, bx, by, ex, ey) in [(30, -60, 110, -110, 150, -180), (38, -20, 130, 0, 190, 40), (30, 20, 110, 100, 140, 190)]:
            d.line([T(side * ax, ay), T(side * bx, by), T(side * ex, ey)], fill=legc, width=int(10 * scale), joint="curve")
    wa = 8 * math.sin(wing_phase)
    for side in (-1, 1):
        wing = Image.new("RGBA", (W, W), (0, 0, 0, 0)); wd = ImageDraw.Draw(wing)
        a, b = T(side * 20 - 90, -40), T(side * 20 + 90, 260)
        wd.ellipse([min(a[0], b[0]), a[1], max(a[0], b[0]), b[1]], fill=(190, 205, 220, 70), outline=(220, 230, 240, 160), width=int(4 * scale))
        wing = wing.rotate(side * (22 + wa), center=T(0, -30), resample=Image.BICUBIC)
        im.alpha_composite(wing)
    ell(-55, 40, 55, 250, fill=(176, 140, 90))                    # abdomen
    for k in range(4):
        y = 100 + k * 36; ell(-50 + k * 3, y, 50 - k * 3, y + 18, fill=(70, 50, 35))
    ell(-62, -70, 62, 70, fill=(150, 115, 72))                    # thorax
    ell(-50, -150, 50, -70, fill=(160, 120, 75))                  # head
    ell(-58, -148, -18, -92, fill=(185, 30, 30)); ell(18, -148, 58, -92, fill=(185, 30, 30))   # eyes
    return im.convert("RGB").resize((S, S), Image.LANCZOS)

# ---------------------------------------------------------------- segments (each yields frames; t = local seconds)
def seg_hook(dur=3.4):
    arms = [("real", "real fly brain", BLUE), ("shuf0", "scrambled fly brain", ORANGE), ("noconn", "no fly brain", GREEN)]
    speed = 2.5; n = int(dur * FPS)
    for k in range(n):
        t = k / FPS; im = canvas(); d = ImageDraw.Draw(im)
        tic = int((0.8 + t) * 35 * speed)
        caption(im, "we scrambled the fly brain.", 28, t + 1, size=58, per_word=0)
        caption(im, "it still plays doom.", 100, t - 0.5, size=58, color=(255, 214, 90), per_word=0.08)
        y = 190
        for a, lab, col in arms:
            fr, kl = DTC[a]["frames"], DTC[a]["kills"]; i = min(tic, len(fr) - 1)
            im.paste(panel(fr[i], 380, 285), (36, y))
            d.rectangle([36, y, 42, y + 285], fill=col)
            d.text((450, y + 40), lab, font=F(40), fill=col, anchor="lm")
            d.text((450, y + 120), "kills", font=F(30, False), fill=INK2, anchor="lm")
            d.text((1040, y + 120), f"{int(kl[i])}", font=F(96), fill=INK, anchor="rm")
            d.text((450, y + 215), f"average over 100 games: {KILLS1[a]:.1f}", font=F(30, False), fill=INK2, anchor="lm")
            y += 295
        yield im

def seg_context(dur=7.2):
    n = int(dur * FPS); act = Activity(SP_REAL)
    doom_fr = DTC["real"]["frames"][300]
    for k in range(n):
        t = k / FPS; im = canvas()
        if t < 2.2:                                        # fly, then zoom toward the head
            f = ease((t - 1.1) / 1.0); z = 1.0 + 4.0 * f * f
            fly = draw_fly(scale=1.6 * z, cy=int(600 + (110 * 1.6 * z + 540 - 600) * f), wing_phase=t * 40)
            im.paste(fly, (0, 0))
            if t > 1.5:
                b = brain(S, S - 120, 0.0, n_show=int(len(Q) * 0.03), zoom=1.9, cz=-0.28); bb = canvas(); bb.paste(b, (0, 110)); im = Image.blend(im, bb, clamp01((t - 1.5) / 0.7))
            small(im, "illustration", (S - 90, S - 40), 20)
            caption(im, "sept 3: scientists mapped", 40, t - 0.1, size=56)
            caption(im, "every connection", 112, t - 0.35, size=56)
            caption(im, "in a fruit fly's brain", 184, t - 0.6, size=56, color=(255, 214, 90))
        elif t < 4.6:                                     # point cloud reveal + count-up
            u = (t - 2.2) / 1.8; nshow = int(len(Q) * ease(u)); th = 0.4 * (t - 2.2)
            a = act.advance_to(int((t - 3.2) * 35)) if t > 3.2 else None
            zz = 1.9 - 0.9 * ease((t - 3.4) / 1.2); im.paste(brain(S, S - 120, th, act=a, n_show=nshow, zoom=zz, cz=-0.28 * (zz - 1) / 0.9), (0, 110))
            cnt = int(N_NEURONS * ease(u))
            ImageDraw.Draw(im).text((S / 2, 70), f"{cnt:,} neurons", font=F(78), fill=INK, anchor="mm")
            small(im, "dots: real positions of neurons in the MaleCNS map", (S / 2, S - 70), 24)
            if t > 3.2: small(im, "lit dots: simulated neuron activity", (S / 2, S - 38), 24, color=BLUE)
        else:                                              # "days later, this went viral"
            u = t - 4.6
            if u < 1.0:
                a = act.advance_to(int((t - 3.2) * 35))
                im.paste(brain(S, S - 120, 0.4 * (t - 2.2), act=a), (0, 110))
                im = Image.blend(im, canvas(), 0.35)
                caption(im, "days later, this went viral:", 470, u, size=64)
            else:
                flash = clamp01(1 - (u - 1.0) / 0.25)
                p = panel(doom_fr, 960, 720); im.paste(p, (60, 230))
                if flash > 0: im = Image.blend(im, Image.new("RGB", (S, S), (255, 255, 255)), flash * 0.8)
                caption(im, "“the fly brain plays doom”", 70, u - 1.0, size=64, color=(255, 214, 90))
        yield im

def seg_question(dur=4.4):
    n = int(dur * FPS)
    for k in range(n):
        t = k / FPS; im = canvas()
        morph = ease((t - 2.4) / 1.2) if t > 2.4 else 0.0
        b = brain(S, S - 200, 0.5 + 0.3 * t, morph=morph, base=0.7)
        im.paste(b, (0, 200))
        if t < 2.2:
            caption(im, "but is the fly brain", 50, t, size=66)
            caption(im, "actually doing the playing?", 135, t - 0.35, size=66, color=(255, 214, 90))
        else:
            caption(im, "the test: break it,", 50, t - 2.2, size=66)
            caption(im, "and see if it still plays.", 135, t - 2.6, size=66, color=(255, 214, 90))
        if morph > 0: small(im, "illustration", (S - 90, S - 40), 20)
        yield im

def box(d, x, y, w, h, title, sub, col, fill=(28, 28, 26), dashed=False):
    if dashed:
        for xx in range(x, x + w, 24): d.line([xx, y, min(xx + 12, x + w), y], fill=col, width=3); d.line([xx, y + h, min(xx + 12, x + w), y + h], fill=col, width=3)
        for yy in range(y, y + h, 24): d.line([x, yy, x, min(yy + 12, y + h)], fill=col, width=3); d.line([x + w, yy, x + w, min(yy + 12, y + h)], fill=col, width=3)
    else:
        d.rounded_rectangle([x, y, x + w, y + h], radius=18, fill=fill, outline=col, width=4)
    if title: d.text((x + w / 2, y + h / 2 - (14 if sub else 0)), title, font=F(34), fill=INK, anchor="mm")
    if sub: d.text((x + w / 2, y + h / 2 + 24), sub, font=F(24, False), fill=INK2, anchor="mm")

def wiring_icon(im, x, y, w, h, kind, seed=3):
    d = ImageDraw.Draw(im); r = np.random.default_rng(seed)
    pts = [(x + 20 + (w - 40) * (i % 6) / 5, y + 18 + (h - 36) * (i // 6) / 2) for i in range(18)]
    if kind == "real":
        for i, p in enumerate(pts):
            for j in (i + 1, i + 6, i + 7):
                if j < 18 and (j != i + 1 or (i % 6) != 5) and not (j == i + 7 and (i % 6) == 5):
                    d.line([p, pts[j]], fill=BLUE, width=2)
        col = BLUE
    elif kind == "shuf":
        for i in range(30):
            a, b = r.integers(0, 18, 2); d.line([pts[a], pts[b]], fill=ORANGE, width=2)
        col = ORANGE
    else:
        return
    for p in pts: d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=col)

def seg_fly1(dur=14.6):
    n = int(dur * FPS); game = DTC["real"]["frames"]
    for k in range(n):
        t = k / FPS; im = canvas(); d = ImageDraw.Draw(im)
        if t < 5.0:                                       # the sandwich
            caption(im, "version 1 is a sandwich.", 40, t, size=62)
            bx, bw = 250, 580
            lay = [(260, "ai eye", "looks at the screen", (160, 160, 160)), (470, "fly brain wiring", "", BLUE), (680, "ai hand", "presses buttons", (160, 160, 160))]
            for i, (y, ti, su, col) in enumerate(lay):
                p = ease((t - 0.4 - 0.45 * i) / 0.4)
                if p <= 0: continue
                yy = int(y - 60 * (1 - p))
                box(d, bx, yy, bw, 170, ti if i != 1 else "", su, col)
                if i == 1:
                    wiring_icon(im, bx + 30, yy + 50, bw - 60, 110, "real"); d.text((bx + bw / 2, yy + 26), ti, font=F(32), fill=INK, anchor="mm")
            if t > 1.0:
                im.paste(panel(game[int(t * 40) % len(game)], 160, 120), (60, 285))
                d.line([225, 345, 245, 345], fill=MUTED, width=3)
            if t > 2.2:
                caption(im, "the whole thing learns", 870, t - 2.2, size=52, color=(255, 214, 90))
                caption(im, "from an expert player.", 935, t - 2.5, size=52, color=(255, 214, 90))
        elif t < 9.4:                                     # swap the middle three ways
            u = t - 5.0
            caption(im, "now swap the middle:", 40, u, size=62)
            cols = [(40, "real", "real fly wiring", BLUE), (380, "shuf", "wires reconnected\nat random", ORANGE), (720, "none", "middle removed", GREEN)]
            for i, (x, kind, lab, col) in enumerate(cols):
                p = ease((u - 0.3 - 0.5 * i) / 0.4)
                if p <= 0: continue
                w = 320; y0 = 200 + int(40 * (1 - p))
                box(d, x, y0, w, 120, "ai eye", "", (150, 150, 150))
                if kind == "none": box(d, x, y0 + 140, w, 170, "", "", col, dashed=True); d.text((x + w / 2, y0 + 225), "(nothing)", font=F(28, False), fill=col, anchor="mm")
                else: box(d, x, y0 + 140, w, 170, "", "", col); wiring_icon(im, x + 20, y0 + 165, w - 40, 120, kind, seed=5 + i)
                box(d, x, y0 + 330, w, 120, "ai hand", "", (150, 150, 150))
                for li, line in enumerate(lab.split("\n")):
                    d.text((x + w / 2, y0 + 500 + li * 38), line, font=F(32), fill=col, anchor="mm")
            if u > 2.4: caption(im, "each one trained the same way", 900, u - 2.4, size=44, color=INK2)
        elif t < 12.4:                                    # bars grow
            u = t - 9.4
            caption(im, "kills per game, 100 games each", 40, u, size=50)
            bars = [("real fly\nwiring", "real", BLUE), ("reconnected\nat random", "shuf0", ORANGE), ("middle\nremoved", "noconn", GREEN)]
            base_y, top_y = 860, 260; vmax = 22.0
            for i, (lab, a, col) in enumerate(bars):
                x = 150 + i * 290; p = ease((u - 0.2 - 0.15 * i) / 1.3); v = KILLS1[a] * p
                h = (base_y - top_y) * v / vmax
                d.rounded_rectangle([x, base_y - h, x + 200, base_y], radius=8, fill=col)
                d.text((x + 100, base_y - h - 50), f"{v:.1f}", font=F(64), fill=INK, anchor="mm")
                for li, line in enumerate(lab.split("\n")):
                    d.text((x + 100, base_y + 45 + li * 38), line, font=F(32), fill=col, anchor="mm")
            d.line([110, base_y, 970, base_y], fill=MUTED, width=2)
        else:                                             # punch
            u = t - 12.4
            caption(im, "same score.", 330, u, size=96)
            caption(im, "the ai parts do the playing.", 500, u - 0.5, size=66, color=(255, 214, 90))
        yield im

def seg_fly2(dur=17.2):
    n = int(dur * FPS)
    act_r = Activity(SP_REAL); act_s = Activity(SP_SHUF)
    shuf_fr, shuf_k = SP_SHUF["frames"], SP_SHUF["kills"]
    for k in range(n):
        t = k / FPS; im = canvas(); d = ImageDraw.Draw(im)
        if t < 2.8:                                       # open loop
            b = brain(S, S - 260, 0.3 * t, base=0.45); im.paste(b, (0, 260))
            caption(im, "but there's a second doom fly.", 40, t, size=58)
            caption(im, "version 2 has no ai at all.", 130, t - 0.7, size=58, color=(255, 214, 90))
            caption(im, "just 166,700 simulated fly neurons.", 210, t - 1.2, size=46, color=INK2)
        elif t < 7.8:                                     # real fly + real spikes, in sync (1x game speed)
            u = t - 2.8; tic = int(u * 35)
            a = act_r.advance_to(tic + 1)
            im.paste(brain(560, 780, 0.25 * u, act=a, color=BLUE, base=0.45), (520, 190))
            im.paste(panel(REAL_FR[tic], 500, 375), (24, 330))
            d.rectangle([24, 322, 524, 328], fill=BLUE)
            d.text((24, 290), "the fly's game", font=F(30), fill=BLUE, anchor="lm")
            d.text((524, 740), f"kills {int(REAL_K[tic])}", font=F(40), fill=INK, anchor="rm")
            small(im, "simulated neuron activity (real run, real time)", (800, 985), 24, color=BLUE)
            caption(im, "its 166,700 neurons, firing live", 40, u, size=54)
        elif t < 11.6:                                    # scrambled: dies
            u = t - 7.8; tic = min(int(u * 35 * 1.6), len(shuf_fr) - 1); dead = tic >= len(shuf_fr) - 1
            a = act_s.advance_to(tic + 1)
            im.paste(brain(560, 780, 0.25 * u, act=a, color=ORANGE, base=0.45), (520, 190))
            p = panel(shuf_fr[tic], 500, 375)
            if dead: p = Image.blend(p, Image.new("RGB", p.size, (0, 0, 0)), 0.6)
            im.paste(p, (24, 330)); d.rectangle([24, 322, 524, 328], fill=ORANGE)
            d.text((24, 290), "scrambled wiring", font=F(30), fill=ORANGE, anchor="lm")
            d.text((524, 740), f"alive {tic / 35:.1f} s", font=F(40), fill=INK, anchor="rm")
            if dead: d.text((274, 517), "dead", font=F(70), fill=INK, anchor="mm")
            small(im, "simulated neuron activity (real run, 1.6x)", (800, 985), 24, color=ORANGE)
            caption(im, "scramble its wiring:", 40, u, size=58)
            if u > 1.6:
                d.rectangle([0, 870, S, 960], fill=BG)
                caption(im, f"dies in {SHUF2['survival_mean']:.0f} s instead of {REAL2['survival_mean']:.0f} s", 880, u - 1.6, size=54, color=(255, 214, 90))
        elif t < 13.6:
            u = t - 11.6
            b = brain(S, S - 300, 0.3 * t, base=0.4); im.paste(b, (0, 300))
            caption(im, "so the brain matters?", 60, u, size=66)
            caption(im, "watch what it actually does:", 160, u - 0.8, size=58, color=(255, 214, 90))
        else:                                             # kinetic text over the real fly, with its real commands
            u = t - 13.6; tic = min(int((u + 6) * 35 * 2), len(REAL_FR) - 1)
            im.paste(panel(REAL_FR[tic], 720, 540), (180, 200))
            words = [("turn right.", 0.0), ("walk.", 0.9), ("shoot.", 1.8), ("repeat.", 2.7)]
            for w, t0 in words:
                if u >= t0:
                    p = ease((u - t0) / 0.2); sz = int(90 * (0.6 + 0.4 * p))
                    col = (255, 214, 90) if w == "repeat." else INK
                    yy = 60 if w != "repeat." else 60
                    if w == words[min(int(u / 0.9), 3)][0]:
                        d.text((S / 2 + 3, 113), w, font=F(sz), fill=(0, 0, 0), anchor="mm"); d.text((S / 2, 110), w, font=F(sz), fill=col, anchor="mm")
            tr, fw, at = REAL_CTRL[max(tic - 1, 0)]
            y = 780
            for lab, val, vmax in [("turn right", tr, 2.0), ("walk", fw, 20.0), ("trigger", at, 1.0)]:
                d.text((180, y), lab, font=F(30), fill=INK2, anchor="lm")
                d.rectangle([420, y - 14, 900, y + 14], fill=(40, 40, 38)); d.rectangle([420, y - 14, 420 + int(480 * clamp01(val / vmax)), y + 14], fill=BLUE)
                y += 55
            small(im, "the fly's actual commands, moment by moment", (S / 2, 960), 24, color=BLUE)
        yield im

def seg_robot(dur=6.4):
    n = int(dur * FPS)
    for k in range(n):
        t = k / FPS; im = canvas(); d = ImageDraw.Draw(im)
        tic = min(int((t + 1) * 35 * 3), len(REAL_FR) - 1, len(ROB_FR) - 1)
        for x, fr, kl, lab, col in [(20, REAL_FR, REAL_K, "fly brain", BLUE), (550, ROB_FR, ROB_K, "robot, no brain", GREEN)]:
            im.paste(panel(fr[tic], 510, 382), (x, 300)); d.rectangle([x, 292, x + 510, 298], fill=col)
            d.text((x, 262), lab, font=F(34), fill=col, anchor="lm")
            d.text((x + 510, 720), f"kills {int(kl[tic])}", font=F(38), fill=INK, anchor="rm")
        caption(im, "a robot doing just those 3 moves:", 40, t, size=52)
        caption(im, "survives just as long. no brain.", 130, t - 1.0, size=52, color=(255, 214, 90))
        if t > 2.4:
            p = ease((t - 2.4) / 1.2)
            for x, st, col in [(20, REAL2, BLUE), (550, ROBOT2, GREEN)]:
                d.text((x + 255, 815), f"{st['survival_mean'] * p:.0f} s alive", font=F(56), fill=col, anchor="mm")
                d.text((x + 255, 885), f"{st['kills_mean'] * p:.0f} kills", font=F(56), fill=col, anchor="mm")
            small(im, f"average of {REAL2['n']} rounds each (60 s max)", (S / 2, 950), 24)
        yield im

REFS = ("wiring map: MaleCNS v1.0 (Janelia / Google) · doom flies: doomfly-rl by nonatofabio, who ran a smaller "
        "version of this test first, and nftechie/doomfly · 100 games per setup · fly drawing: own illustration · "
        "code: github.com/gabrycina/doom-fly-control")

def seg_end(dur=4.6):
    n = int(dur * FPS)
    for k in range(n):
        t = k / FPS; im = canvas(); d = ImageDraw.Draw(im)
        caption(im, "the fly brain", 320, t, size=100)
        caption(im, "isn't playing doom.", 450, t - 0.4, size=100, color=(255, 214, 90))
        if t > 1.4:
            f = F(22, False); y = 900
            for line in wrap(d, REFS, f, 980): d.text((S / 2, y), line, font=f, fill=MUTED, anchor="mm"); y += 30
        yield im

def main():
    out = ROOT / "demo/demo.mp4"
    segs = [seg_hook, seg_context, seg_question, seg_fly1, seg_fly2, seg_robot, seg_end]
    total = sum(int(f.__defaults__[0] * FPS) for f in segs)
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{S}x{S}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "slow", "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                         stdin=subprocess.PIPE)
    n = 0; marks = []
    for g in segs:
        marks.append((g.__name__, n / FPS))
        for im in g():
            progress(im, n / total)
            p.stdin.write(im.tobytes()); n += 1
            if n == 45: im.save(ROOT / "demo/poster.png")
    p.stdin.close(); p.wait()
    print(json.dumps(marks), n / FPS, "s", out.stat().st_size / 1e6, "MB")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:   # preview: segment name + local time -> png
        g = globals()[sys.argv[1]]; tt = float(sys.argv[2])
        for i, im in enumerate(g()):
            if i == int(tt * FPS): im.save(sys.argv[3]); break
    else:
        main()
