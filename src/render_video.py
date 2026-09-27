"""Compose demo.mp4 (1920x1080, 30 fps, H.264 yuv420p) from recorded footage + charts."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[1]
import imageio_ffmpeg; FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1920, 1080, 30
BG = (17, 17, 16); INK = (255, 255, 255); INK2 = (195, 194, 183); MUTED = (138, 137, 128)
COL = {"real": (57, 135, 229), "shuf0": (217, 89, 38), "noconn": (25, 158, 112)}
B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"; R = "/System/Library/Fonts/Supplemental/Arial.ttf"
def F(sz, bold=False): return ImageFont.truetype(B if bold else R, sz)
CFG = json.loads((ROOT / "video/captions.json").read_text())

def text_c(d, xy, s, font, fill, anchor="mm"): d.text(xy, s, font=font, fill=fill, anchor=anchor)

def wrap(d, s, font, maxw):
    words, lines, cur = s.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= maxw: cur = t
        else: lines.append(cur); cur = w
    return lines + [cur]

def header(d, title, sub, y=62):
    tl = wrap(d, title, F(52, True), 1820)
    for i, line in enumerate(tl): text_c(d, (W // 2, y + i * 60), line, F(52, True), INK)
    y += 60 * (len(tl) - 1)
    for i, line in enumerate(wrap(d, sub, F(30), 1780)):
        text_c(d, (W // 2, y + 62 + i * 38), line, F(30), INK2)

def footer(d, s, y=1040):
    for i, line in enumerate(wrap(d, s, F(24), 1840)[::-1]):
        text_c(d, (W // 2, y - i * 32), line, F(24), MUTED)

def seg_doom():
    arms = ["real", "shuf0", "noconn"]
    clips = {a: np.load(ROOT / f"footage/{a}_dtc_ep0.npz") for a in arms}
    avg = CFG["doom_avg"]
    speed = CFG.get("doom_speed", 2.5)
    n_out = int(max(len(clips[a]["frames"]) for a in arms) / 35 / speed * FPS) + FPS  # + 1 s hold
    pw, ph = 600, 450; xs = [30, 660, 1290]; py = 300
    for k in range(n_out):
        im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
        header(d, CFG["doom_title"], CFG["doom_sub"])
        tic = int(k / FPS * 35 * speed)
        for a, x in zip(arms, xs):
            fr, kills = clips[a]["frames"], clips[a]["kills"]
            i = min(tic, len(fr) - 1); dead = tic >= len(fr) - 1
            p = Image.fromarray(fr[i]).resize((pw, ph), Image.NEAREST)
            if dead: p = Image.blend(p, Image.new("RGB", p.size, (0, 0, 0)), 0.55)
            im.paste(p, (x, py))
            d.rectangle([x, py - 8, x + pw, py - 2], fill=COL[a])
            text_c(d, (x, py - 68), CFG["labels"][a][0], F(34, True), INK, "lm")
            text_c(d, (x, py - 30), CFG["labels"][a][1], F(24), INK2, "lm")
            if dead: text_c(d, (x + pw // 2, py + ph // 2), "died", F(48, True), INK)
            text_c(d, (x, py + ph + 42), "kills this game", F(28), INK2, "lm")
            text_c(d, (x + pw, py + ph + 42), f"{int(kills[i])}", F(56, True), INK, "rm")
            d.line([x, py + ph + 84, x + pw, py + ph + 84], fill=(44, 44, 42), width=2)
            m, lo, hi = avg[a]
            text_c(d, (x, py + ph + 124), "average over 100 games", F(28), INK2, "lm")
            text_c(d, (x + pw, py + ph + 124), f"{m:.1f}", F(56, True), COL[a], "rm")
            text_c(d, (x + pw, py + ph + 170), f"95% CI {lo:.1f} to {hi:.1f}", F(22), MUTED, "rm")
        footer(d, CFG["doom_footer"])
        yield im

def seg_duo():
    c = CFG["duo"]; arms = c["arms"]
    clips = {a: np.load(ROOT / f"footage/a2/{a}_ep0.npz") for a in arms}
    speed = c["speed"]; secs = c["game_secs"] / speed
    pw, ph = 880, 660; xs = [60, 980]; py = 250
    for k in range(int(secs * FPS)):
        im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
        header(d, c["title"], c["sub"])
        tic = int(k / FPS * 35 * speed)
        for a, x, lab, col in zip(arms, xs, c["labels"], c["colors"]):
            fr, kills = clips[a]["frames"], clips[a]["kills"]
            i = min(tic, len(fr) - 1)
            im.paste(Image.fromarray(fr[i]).resize((pw, ph - 60), Image.BILINEAR), (x, py + 60))
            d.rectangle([x, py + 50, x + pw, py + 56], fill=tuple(col))
            text_c(d, (x, py + 20), lab, F(34, True), INK, "lm")
            text_c(d, (x + pw, py + 20), f"kills {int(kills[i])}", F(34, True), INK, "rm")
        text_c(d, (W // 2, py + ph + 40), f"game time {tic / 35:4.1f} s  ({speed:g}x speed)", F(24), MUTED)
        footer(d, c["footer"])
        yield im

def seg_static(img_path, title, sub, foot, secs, chart_y=250):
    ch = Image.open(img_path).convert("RGB")
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    header(d, title, sub); im.paste(ch, (0, chart_y)); footer(d, foot)
    for _ in range(int(secs * FPS)): yield im

def seg_card(lines, foot, secs):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    y = 330
    for s, sz, bold, col in lines:
        for line in wrap(d, s, F(sz, bold), 1700):
            text_c(d, (W // 2, y), line, F(sz, bold), tuple(col)); y += int(sz * 1.35)
        y += 26
    footer(d, foot)
    for _ in range(int(secs * FPS)): yield im

def fade(gen_a_last, gen_b_first, n=8):
    for t in range(1, n + 1): yield Image.blend(gen_a_last, gen_b_first, t / (n + 1))

def main():
    out = ROOT / "demo/demo.mp4"; out.parent.mkdir(exist_ok=True)
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "slow", "-crf", str(CFG.get("crf", 21)), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                          "-r", str(FPS), str(out)], stdin=subprocess.PIPE)
    st = [seg_static(ROOT / s["img"], s["title"], s["sub"], s["footer"], s["secs"]) for s in CFG["static"]]
    segs = [seg_doom(), st[0], seg_duo(), st[1], st[2]] \
           + [seg_card(CFG["end"]["lines"], CFG["end"]["footer"], CFG["end"]["secs"])]
    last = None; poster_done = False; n = 0
    for g in segs:
        first = True
        for im in g:
            if first and last is not None:
                for f in fade(last, im): p.stdin.write(f.tobytes()); n += 1
            first = False
            p.stdin.write(im.tobytes()); n += 1; last = im
            if not poster_done and n == CFG.get("poster_frame", 150):
                im.save(ROOT / "demo/poster.png"); poster_done = True
    p.stdin.close(); p.wait()
    print(out, n / FPS, "s", out.stat().st_size / 1e6, "MB")

if __name__ == "__main__": main()
