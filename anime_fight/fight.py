"""
KAITO vs RYUGA - animatie de lupta in stil anime (30 sec), generata procedural.
Personaje originale:
  - Kaito: par turcoaz tepos, esarfa rosie, stapaneste fulgerul (Sfera Tunetului)
  - Ryuga: par lung negru, mantie intunecata, stapaneste focul (Dragonul Stacojiu)

Rulare:  python3 fight.py   -> produce kaito_vs_ryuga.mp4
Necesita: Pillow, numpy, ffmpeg
"""
import math
import os
import random
import subprocess
import sys
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 720
FPS = 24
DUR = 30.0
N = int(FPS * DUR)
GROUND = 560
S = 1.25  # scara personajelor
BLACK = (15, 12, 22)
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"
FONT_X = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
_fonts = {}


def font(size, heavy=False):
    k = (size, heavy)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT_X if heavy else FONT, size)
    return _fonts[k]


# ----------------------------------------------------------------- utilitare
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def lerp(a, b, t):
    return a + (b - a) * t


def ease(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def seg(t, a, b):
    return clamp((t - a) / (b - a))


def mixc(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def dark(c, k=0.72):
    return tuple(int(v * k) for v in c[:3]) + tuple(c[3:])


def circ(d, p, r, fill, outline=None, width=1):
    d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=fill, outline=outline, width=width)


class Cam:
    def __init__(self, cx=640, cy=360, z=1.0, sx=0.0, sy=0.0):
        self.cx, self.cy, self.z, self.sx, self.sy = cx, cy, z, sx, sy

    def p(self, x, y, par=1.0):
        zz = 1 + (self.z - 1) * par
        cx = 640 + (self.cx - 640) * par
        cy = 360 + (self.cy - 360) * par
        return ((x - cx) * zz + 640 + self.sx * par, (y - cy) * zz + 360 + self.sy * par)


# ----------------------------------------------------------------- poze
POSES = {
    "stance":   dict(lean=10, head=0, ub=25, fb=110, uf=45, ff=130, tb=-25, sb=-5, tf=30, sf=0),
    "run1":     dict(lean=45, head=-15, ub=-120, fb=-100, uf=-110, ff=-95, tb=-50, sb=-100, tf=70, sf=10),
    "run2":     dict(lean=45, head=-15, ub=-115, fb=-95, uf=-120, ff=-100, tb=60, sb=0, tf=-40, sf=-90),
    "crun1":    dict(lean=40, head=-15, ub=-110, fb=-90, uf=80, ff=85, tb=-50, sb=-100, tf=70, sf=10),
    "crun2":    dict(lean=40, head=-15, ub=-110, fb=-90, uf=80, ff=85, tb=60, sb=0, tf=-40, sf=-90),
    "punch":    dict(lean=25, head=-5, ub=-20, fb=70, uf=90, ff=90, tb=-45, sb=-30, tf=50, sf=10),
    "kick":     dict(lean=-25, head=-10, ub=-60, fb=20, uf=60, ff=150, tb=-10, sb=0, tf=105, sf=100),
    "guard":    dict(lean=0, head=10, ub=30, fb=165, uf=40, ff=170, tb=-20, sb=-5, tf=25, sf=5),
    "hit":      dict(lean=-35, head=-30, ub=-80, fb=-120, uf=140, ff=170, tb=-30, sb=-60, tf=40, sf=-20),
    "handsign": dict(lean=5, head=5, ub=35, fb=150, uf=40, ff=150, tb=-22, sb=-5, tf=28, sf=2),
    "charge":   dict(lean=25, head=-5, ub=60, fb=85, uf=85, ff=88, tb=-45, sb=-30, tf=45, sf=5),
    "beam":     dict(lean=15, head=-5, ub=80, fb=88, uf=85, ff=90, tb=-40, sb=-25, tf=40, sf=5),
    "kneel":    dict(lean=35, head=20, ub=10, fb=30, uf=60, ff=10, tb=5, sb=-90, tf=90, sf=0),
    "jump":     dict(lean=15, head=0, ub=-40, fb=40, uf=60, ff=140, tb=40, sb=-40, tf=90, sf=10),
    "flykick":  dict(lean=-10, head=-5, ub=-100, fb=-80, uf=-90, ff=-70, tb=-20, sb=-60, tf=95, sf=95),
    "tired":    dict(lean=30, head=15, ub=5, fb=20, uf=15, ff=40, tb=-20, sb=-5, tf=25, sf=0),
}


def resolve(name, t):
    if name == "run":
        p = POSES["run1" if int(t * 10) % 2 == 0 else "run2"]
    elif name == "crun":
        p = POSES["crun1" if int(t * 10) % 2 == 0 else "crun2"]
    elif name == "flurry":
        p = POSES[["punch", "kick", "guard", "punch", "kick"][int(t * 12) % 5]]
    else:
        p = POSES[name]
    p = dict(p)
    if name in ("stance", "tired", "kneel", "handsign"):
        amp = 4 if name == "tired" or name == "kneel" else 2
        p["lean"] += amp * math.sin(t * (5 if name != "stance" else 3))
    return p


def blend(p1, p2, u):
    return {k: lerp(p1[k], p2[k], u) for k in p1}


def vec(a, f):
    r = math.radians(a)
    return (f * math.sin(r), math.cos(r))


LEN = dict(torso=58 * S, ua=30 * S, fa=28 * S, th=40 * S, sh=40 * S, head=17 * S, neck=8 * S)


def joints_rel(p, f):
    tv = vec(180 - p["lean"], f)
    neck = (tv[0] * LEN["torso"], tv[1] * LEN["torso"])
    sh = (neck[0] * 0.92, neck[1] * 0.92)
    hv = vec(180 - p["lean"] - p["head"], f)
    hd = LEN["neck"] + LEN["head"]
    head = (neck[0] + hv[0] * hd, neck[1] + hv[1] * hd)

    def limb(base, a1, a2, l1, l2):
        v1 = vec(a1, f)
        m = (base[0] + v1[0] * l1, base[1] + v1[1] * l1)
        v2 = vec(a2, f)
        return m, (m[0] + v2[0] * l2, m[1] + v2[1] * l2)

    eb, hb = limb(sh, p["ub"], p["fb"], LEN["ua"], LEN["fa"])
    ef, hf = limb(sh, p["uf"], p["ff"], LEN["ua"], LEN["fa"])
    kb, fb = limb((0, 0), p["tb"], p["sb"], LEN["th"], LEN["sh"])
    kf, ff = limb((0, 0), p["tf"], p["sf"], LEN["th"], LEN["sh"])
    return dict(hip=(0, 0), neck=neck, sh=sh, head=head, eb=eb, hb=hb, ef=ef, hf=hf,
                kb=kb, fb=fb, kf=kf, ff=ff, hv=hv, tv=tv)


def joints_world(x, air, p, f):
    j = joints_rel(p, f)
    maxy = max(j["fb"][1], j["ff"][1], j["kb"][1], j["kf"][1])
    hy = GROUND - maxy - 5 * S - air
    return {k: ((v[0] + x, v[1] + hy) if k not in ("hv", "tv") else v) for k, v in j.items()}


# ----------------------------------------------------------------- personaje
KAITO = dict(name="KAITO", skin=(250, 214, 182), hair=(45, 205, 215), top=(238, 238, 248),
             top2=(40, 95, 205), pants=(38, 48, 92), scarf=(225, 40, 55), band=(55, 65, 95),
             eye=(60, 170, 255), kind="spiky", aura=(80, 170, 255))
RYUGA = dict(name="RYUGA", skin=(232, 198, 172), hair=(42, 26, 50), top=(58, 52, 70),
             top2=(195, 30, 45), pants=(34, 30, 42), scarf=None, band=(125, 125, 140),
             eye=(240, 30, 45), kind="long", aura=(255, 90, 30))


def limb_draw(d, pts, col, w, ol):
    wi = max(1, int(w))
    for a, b in zip(pts, pts[1:]):
        d.line([a, b], fill=ol, width=wi + 5)
    for p in pts:
        circ(d, p, (wi + 5) / 2, ol)
    for a, b in zip(pts, pts[1:]):
        d.line([a, b], fill=col, width=wi)
    for p in pts:
        circ(d, p, wi / 2, col)


def ribbon(d, cam, base, dirx, diry, length, w0, w1, t, freq, amp, col, ol, n=10, phase=0.0):
    up, lo = [], []
    for i in range(n + 1):
        s = i / n
        x = base[0] + dirx * length * s
        y = base[1] + diry * length * s + math.sin(t * freq - s * 5 + phase) * amp * s
        w = lerp(w0, w1, s)
        up.append(cam.p(x, y - w))
        lo.append(cam.p(x, y + w))
    d.polygon(up + lo[::-1], fill=col, outline=ol, width=max(1, int(2 * cam.z)))


def draw_fighter(d, cam, x, air, p, f, st, t, tint=None):
    """Deseneaza personajul. tint=(r,g,b,a) -> silueta colorata (afterimage)."""
    j = joints_world(x, air, p, f)
    C = (lambda c: tint) if tint else (lambda c: c)
    ol = tint if tint else BLACK
    z = cam.z
    P = lambda q: cam.p(*q)
    hr = LEN["head"]
    hc = j["head"]

    # par lung / mantie / esarfa (in spate)
    if st["kind"] == "long":
        ribbon(d, cam, (hc[0] - f * hr * 0.4, hc[1] - hr * 0.2), -f, 0.45, 85 * S, hr * 0.85, 3,
               t, 9, 10, C(st["hair"]), ol, phase=1.0)
        # mantie
        sh, hip = j["sh"], j["hip"]
        fl = math.sin(t * 10) * 10
        pts = [sh, (sh[0] - f * 10, sh[1] + 5), (hip[0] - f * (45 + fl), hip[1] + 45),
               (hip[0] - f * (20 + fl * 0.5), hip[1] + 62), (hip[0] + f * 5, hip[1] + 40), hip]
        d.polygon([P(q) for q in pts], fill=C((45, 40, 56)), outline=ol, width=max(1, int(2 * z)))
        d.line([P(pts[2]), P(pts[3])], fill=C(st["top2"]), width=max(1, int(4 * z)))
    if st["scarf"]:
        ribbon(d, cam, j["neck"], -f, 0.12, 80 * S, 6 * S, 3 * S, t, 14, 14, C(st["scarf"]), ol)
    # banderola - capetele
    back = (hc[0] - f * hr * 0.95, hc[1] - hr * 0.2)
    ribbon(d, cam, back, -f, 0.25, 26 * S, 2.6 * S, 1.5 * S, t, 16, 6, C(st["band"]), ol, n=6)
    ribbon(d, cam, back, -f, 0.55, 22 * S, 2.4 * S, 1.2 * S, t, 16, 6, C(st["band"]), ol, n=6, phase=2)

    aw, lw = 9 * S * z, 11 * S * z
    # brat spate
    limb_draw(d, [P(j["sh"]), P(j["eb"]), P(j["hb"])], C(dark(st["top"])), aw, ol)
    circ(d, P(j["hb"]), 6 * S * z, C(dark(st["skin"])), ol, max(1, int(2 * z)))
    # picioare
    for k, s in (("b", 0.72), ("f", 1.0)):
        col = st["pants"] if s == 1 else dark(st["pants"], s)
        limb_draw(d, [P(j["hip"]), P(j["k" + k]), P(j["f" + k])], C(col), lw, ol)
        fp = P(j["f" + k])
        d.ellipse([fp[0] - 8 * S * z + f * 4 * z, fp[1] - 4 * S * z, fp[0] + 8 * S * z + f * 4 * z,
                   fp[1] + 4 * S * z], fill=C((30, 28, 34)), outline=ol)
    # tors
    hip, sh = j["hip"], j["sh"]
    dx, dy = sh[0] - hip[0], sh[1] - hip[1]
    ln = math.hypot(dx, dy) or 1
    px, py = -dy / ln, dx / ln
    hw, sw = 13 * S, 16 * S
    tors = [(hip[0] + px * hw, hip[1] + py * hw), (sh[0] + px * sw, sh[1] + py * sw),
            (sh[0] - px * sw, sh[1] - py * sw), (hip[0] - px * hw, hip[1] - py * hw)]
    d.polygon([P(q) for q in tors], fill=C(st["top"]), outline=ol, width=max(1, int(3 * z)))
    side = f if py * f >= 0 else -f
    d.line([P((hip[0] + px * 4 * side, hip[1] + py * 4 * side)), P((sh[0] + px * 6 * side, sh[1] + py * 6 * side))],
           fill=C(st["top2"]), width=max(1, int(4 * z)))
    d.line([P(tors[0]), P(tors[3])], fill=C(st["top2"]), width=max(1, int(6 * z)))
    # cap
    circ(d, P(hc), hr * z, C(st["skin"]), ol, max(1, int(3 * z)))
    if not tint:
        ex, ey = hc[0] + f * hr * 0.45, hc[1] - hr * 0.02
        d.line([P((ex - f * hr * 0.25, ey - hr * 0.2)), P((ex + f * hr * 0.35, ey - hr * 0.02))],
               fill=BLACK, width=max(1, int(3 * z)))
        circ(d, P((ex + f * hr * 0.05, ey + hr * 0.05)), 2.8 * z * S, st["eye"])
        d.line([P((ex - f * hr * 0.3, ey - hr * 0.45)), P((ex + f * hr * 0.4, ey - hr * 0.25))],
               fill=BLACK, width=max(1, int(3 * z)))
        d.line([P((hc[0] + f * hr * 0.45, hc[1] + hr * 0.55)), P((hc[0] + f * hr * 0.8, hc[1] + hr * 0.5))],
               fill=BLACK, width=max(1, int(2 * z)))
    # banderola
    def hp(phi, rr):
        a = math.radians(phi)
        return (hc[0] - f * math.sin(a) * rr, hc[1] - math.cos(a) * rr)
    d.line([P(hp(-72, hr * 1.02)), P(hp(100, hr * 1.02))], fill=C(st["band"]), width=max(1, int(6 * S * z)))
    plate = [hp(-80, hr * 1.05), hp(-55, hr * 1.12), hp(-38, hr * 0.98), hp(-62, hr * 0.85)]
    d.polygon([P(q) for q in plate], fill=C((200, 205, 215)), outline=ol)
    # par
    pts = []
    if st["kind"] == "spiky":
        spikes = 9
        for i in range(spikes + 1):
            phi = lerp(-35, 205, i / spikes)
            pts.append(hp(phi, hr * 0.92))
            if i < spikes:
                ph2 = lerp(-35, 205, (i + 0.5) / spikes) + 25
                pts.append(hp(ph2, hr * (1.75 + 0.15 * math.sin(t * 9 + i))))
    else:
        spikes = 7
        for i in range(spikes + 1):
            phi = lerp(-45, 190, i / spikes)
            pts.append(hp(phi, hr * 0.95))
            if i < spikes:
                ph2 = lerp(-45, 190, (i + 0.5) / spikes) + 15
                pts.append(hp(ph2, hr * (1.3 + 0.06 * math.sin(t * 8 + i))))
    d.polygon([P(q) for q in pts], fill=C(st["hair"]), outline=ol, width=max(1, int(2 * z)))
    # brat fata
    limb_draw(d, [P(j["sh"]), P(j["ef"]), P(j["hf"])], C(st["top"]), aw, ol)
    d.line([P(j["ef"]), P(j["hf"])], fill=C(st["top2"]), width=max(1, int(aw * 0.6)))
    circ(d, P(j["hf"]), 6 * S * z, C(st["skin"]), ol, max(1, int(2 * z)))
    return j


# ----------------------------------------------------------------- coregrafie
K_KEYS = [
    (0, 300, 0, "stance", 1), (6.0, 300, 0, "stance", 1),
    (6.15, 330, 0, "run", 1), (7.15, 600, 0, "run", 1),
    (7.2, 612, 0, "punch", 1), (7.55, 612, 0, "punch", 1),
    (7.9, 530, 0, "guard", 1), (8.5, 520, 0, "stance", 1),
    (8.75, 520, 0, "guard", 1), (9.0, 525, 0, "guard", 1),
    (9.3, 600, 0, "punch", 1), (9.5, 590, 0, "stance", 1),
    (9.8, 620, 160, "jump", 1), (9.95, 630, 180, "flykick", 1),
    (10.1, 625, 190, "guard", 1), (10.35, 650, 200, "kick", 1),
    (10.8, 600, 0, "stance", 1), (11.25, 590, 0, "stance", 1),
    (11.45, 585, 0, "hit", 1), (11.9, 360, 40, "hit", 1),
    (12.2, 310, 0, "guard", 1), (12.5, 320, 0, "stance", 1),
    (12.7, 585, 0, "run", 1), (12.75, 590, 0, "flurry", 1),
    (13.6, 590, 0, "flurry", 1), (13.75, 520, 80, "jump", 1),
    (14.0, 330, 0, "stance", 1), (14.2, 330, 0, "handsign", 1),
    (15.0, 330, 0, "handsign", 1), (15.2, 330, 0, "charge", 1),
    (18.0, 330, 0, "charge", 1), (18.1, 345, 0, "crun", 1),
    (18.75, 560, 0, "crun", 1), (18.8, 560, 0, "charge", 1),
    (21.9, 575, 0, "charge", 1), (23.0, 400, 0, "kneel", 1), (30, 400, 0, "kneel", 1),
]
R_KEYS = [
    (0, 980, 0, "stance", -1), (6.0, 980, 0, "stance", -1),
    (6.15, 950, 0, "run", -1), (7.15, 680, 0, "run", -1),
    (7.2, 668, 0, "punch", -1), (7.55, 668, 0, "punch", -1),
    (7.9, 760, 0, "guard", -1), (8.5, 770, 0, "stance", -1),
    (8.6, 700, 0, "run", -1), (8.85, 650, 0, "kick", -1),
    (9.1, 690, 0, "stance", -1), (9.3, 690, 0, "guard", -1),
    (9.5, 700, 0, "stance", -1), (9.8, 700, 160, "jump", -1),
    (9.95, 690, 180, "guard", -1), (10.1, 680, 190, "punch", -1),
    (10.35, 690, 200, "hit", -1), (10.7, 720, 0, "hit", -1),
    (10.9, 720, 0, "kneel", -1), (11.3, 670, 0, "stance", -1),
    (11.45, 660, 0, "punch", -1), (11.8, 670, 0, "stance", -1),
    (12.5, 680, 0, "stance", -1), (12.75, 685, 0, "flurry", -1),
    (13.6, 685, 0, "flurry", -1), (13.75, 760, 80, "jump", -1),
    (14.0, 950, 0, "stance", -1), (14.4, 950, 0, "handsign", -1),
    (15.4, 950, 0, "handsign", -1), (15.6, 950, 0, "beam", -1),
    (21.9, 950, 0, "beam", -1), (23.0, 880, 0, "tired", -1), (30, 880, 0, "tired", -1),
]
HIDDEN = {"K": [(22.0, 23.6)], "R": [(22.0, 23.6), (10.95, 11.25)]}
AFTER = [(6.15, 7.15), (12.5, 13.7), (18.1, 18.75), (9.75, 10.4)]


def state(keys, t):
    if t <= keys[0][0]:
        k = keys[0]
        return k[1], k[2], resolve(k[3], t), k[4]
    for a, b in zip(keys, keys[1:]):
        if a[0] <= t <= b[0]:
            raw = (t - a[0]) / max(1e-6, b[0] - a[0])
            u = raw if a[3] in ("run", "crun") else ease(raw)
            pu = ease(clamp(raw * 1.6))
            pose = blend(resolve(a[3], t), resolve(b[3], t), pu)
            return lerp(a[1], b[1], u), lerp(a[2], b[2], u), pose, (b[4] if raw > 0.5 else a[4])
    k = keys[-1]
    return k[1], k[2], resolve(k[3], t), k[4]


def hidden(who, t):
    return any(a <= t < b for a, b in HIDDEN[who])


# evenimente: impacturi (t, x, y, marime)
IMPACTS = [(7.2, 640, 395, 2.0), (8.9, 592, 405, 1.0), (9.35, 668, 400, 1.0),
           (9.95, 662, 228, 1.0), (10.1, 648, 215, 1.0), (10.35, 682, 222, 1.6),
           (10.7, 722, 540, 1.6), (11.45, 590, 400, 1.6)]
FLURRY = [12.8 + i * 0.075 for i in range(11)]
_rf = random.Random(5)
for ft in FLURRY:
    IMPACTS.append((ft, 637 + _rf.uniform(-25, 25), 390 + _rf.uniform(-50, 40), 0.6))
DUSTS = [(7.25, 640, GROUND, 1.5), (7.6, 560, GROUND, 0.8), (7.6, 720, GROUND, 0.8),
         (10.72, 722, GROUND, 1.6), (12.2, 300, GROUND, 1.0), (14.0, 330, GROUND, 0.9),
         (14.0, 950, GROUND, 0.9), (18.15, 345, GROUND, 0.9)]
SMOKES = [(10.92, 720, 470), (11.22, 670, 470)]
INVERT = [(7.2, 7.29), (10.35, 10.42), (18.8, 18.88), (21.95, 22.0)]
SHAKES = [(7.2, 0.45, 16), (8.9, 0.15, 5), (9.35, 0.15, 5), (10.35, 0.2, 9), (10.7, 0.35, 12),
          (11.45, 0.3, 10), (12.8, 0.8, 6), (18.8, 3.2, 7), (22.0, 1.8, 28)]
CAM_KEYS = [
    (0, 640, 360, 1.0), (4, 640, 340, 1.12),
    (6, 640, 360, 1.0), (7.0, 640, 380, 1.15), (7.6, 640, 380, 1.28), (8.5, 640, 380, 1.1),
    (9.5, 640, 360, 1.15), (9.8, 650, 290, 1.3), (10.4, 660, 290, 1.3), (10.8, 680, 400, 1.15),
    (11.5, 600, 390, 1.2), (12.2, 480, 390, 1.1), (12.7, 640, 380, 1.3), (13.6, 640, 380, 1.38),
    (14.1, 640, 360, 1.0), (15.0, 640, 360, 1.05), (15.15, 335, 395, 2.0), (16.2, 340, 395, 2.15),
    (16.3, 945, 395, 2.0), (17.3, 940, 395, 2.15), (17.45, 640, 360, 1.0), (18.8, 630, 380, 1.15),
    (21.9, 640, 380, 1.45), (22.0, 640, 360, 1.0), (25, 640, 360, 1.0), (30, 640, 340, 1.1),
]


def camera(t):
    cx, cy, z = CAM_KEYS[-1][1:]
    for a, b in zip(CAM_KEYS, CAM_KEYS[1:]):
        if a[0] <= t <= b[0]:
            u = ease((t - a[0]) / (b[0] - a[0]))
            cx, cy, z = (lerp(a[i], b[i], u) for i in (1, 2, 3))
            break
    sx = sy = 0.0
    for t0, dur, amp in SHAKES:
        if t0 <= t < t0 + dur:
            k = amp * (1 - (t - t0) / dur)
            sx += math.sin(t * 97.3 + t0) * k
            sy += math.cos(t * 83.1 + t0 * 2) * k
    return Cam(cx, cy, z, sx, sy)


# ----------------------------------------------------------------- decor
def _mountains(seed, base, amp, n):
    r = random.Random(seed)
    pts = [(-900, 1400)]
    for i in range(n + 1):
        x = -900 + i * (3100 / n)
        pts.append((x, base - r.uniform(0.2, 1.0) * amp))
    pts.append((2200, 1400))
    return pts


MOUNTAINS = [(_mountains(1, 470, 200, 14), 0.15, (120, 70, 110), (30, 30, 60)),
             (_mountains(2, 505, 140, 20), 0.35, (90, 50, 85), (22, 22, 45)),
             (_mountains(3, 545, 80, 28), 0.6, (60, 35, 60), (15, 15, 32))]
_rr = random.Random(9)
ROCKS = []
for _ in range(14):
    x = _rr.uniform(-300, 1600)
    s = _rr.uniform(10, 34)
    y = GROUND + _rr.uniform(10, 150)
    ROCKS.append([(x - s, y), (x - s * 0.6, y - s * 0.8), (x + s * 0.4, y - s), (x + s, y - s * 0.2), (x + s * 0.8, y)])
STARS = [(random.Random(i).uniform(0, W), random.Random(i * 3).uniform(40, 420), random.Random(i * 7).uniform(0.8, 2.2))
         for i in range(120)]


def make_crack(x, seed, n=7, length=110):
    r = random.Random(seed)
    lines = []
    for b in range(n):
        ang = r.uniform(-0.5, 0.5) + (0 if b % 2 else math.pi)
        px, py = x, GROUND + 2
        pts = [(px, py)]
        for _ in range(5):
            ang += r.uniform(-0.6, 0.6)
            step = length / 5 * r.uniform(0.6, 1.3)
            px += math.cos(ang) * step
            py += abs(math.sin(ang)) * step * 0.25 + 1
            pts.append((px, py))
        lines.append(pts)
    return lines


CRACK1 = make_crack(722, 3)
CRACK2 = make_crack(700, 4, n=9, length=170)


def sky_phase(t):
    return clamp(seg(t, 14.5, 17.5) * 0.65 + seg(t, 22.0, 24.0) * 0.35)


def draw_background(img, cam, t):
    ph = sky_phase(t)
    top = mixc((70, 40, 115), (8, 10, 30), ph)
    mid = mixc((235, 110, 85), (45, 30, 80), ph)
    hor = mixc((255, 185, 100), (80, 50, 100), ph)
    ys = np.linspace(0, 1, H)[:, None]
    a = np.array(top, float)
    b = np.array(mid, float)
    c = np.array(hor, float)
    col = np.where(ys < 0.45, a + (b - a) * (ys / 0.45), b + (c - b) * np.clip((ys - 0.45) / 0.35, 0, 1))
    arr = np.repeat(col[:, None, :], W, axis=1).astype(np.uint8)
    img.paste(Image.fromarray(arr.reshape(H, W, 3), "RGB"))
    d = ImageDraw.Draw(img)
    # stele + luna
    if ph > 0.4:
        sa = seg(ph, 0.4, 1.0)
        for (x, y, r) in STARS:
            tw = 0.6 + 0.4 * math.sin(t * 3 + x)
            circ(d, (x, y), r, mixc(top, (255, 255, 255), sa * tw))
    if t > 22.5:
        ma = seg(t, 23.0, 25.0)
        mp = cam.p(960, 190, 0.05)
        for k in range(5, 0, -1):
            circ(d, mp, 48 + k * 9, mixc(top, (110, 110, 160), ma * 0.25 / k * 3))
        circ(d, mp, 48, mixc(top, (245, 240, 220), ma))
    # soare
    sp = cam.p(640, 390 + ph * 300, 0.1)
    sr = 120 * (1 + (cam.z - 1) * 0.1)
    circ(d, sp, sr * 1.3, mixc(mid, (255, 160, 90), 0.5 * (1 - ph)))
    circ(d, sp, sr, mixc(hor, (255, 225, 150), 1 - ph))
    # munti
    for pts, par, c1, c2 in MOUNTAINS:
        d.polygon([cam.p(x, y, par) for x, y in pts], fill=mixc(c1, c2, ph))
    # sol
    gcol = mixc((95, 60, 55), (30, 24, 34), ph)
    d.polygon([cam.p(-3000, GROUND), cam.p(4000, GROUND), cam.p(4000, 3000), cam.p(-3000, 3000)], fill=gcol)
    d.line([cam.p(-3000, GROUND), cam.p(4000, GROUND)], fill=mixc((170, 100, 80), (60, 45, 70), ph), width=max(1, int(4 * cam.z)))
    for i in range(6):
        y = GROUND + 18 + i * i * 9
        d.line([cam.p(-3000, y), cam.p(4000, y)], fill=dark(gcol, 0.88), width=max(1, int(2 * cam.z)))
    for rk in ROCKS:
        d.polygon([cam.p(*q) for q in rk], fill=dark(gcol, 0.75), outline=BLACK)
    # crater
    if t > 22.2:
        cr = ease(seg(t, 22.2, 22.6))
        c0 = cam.p(640, GROUND + 25)
        rw, rh = 330 * cr * cam.z, 45 * cr * cam.z
        d.ellipse([c0[0] - rw - 15, c0[1] - rh - 8, c0[0] + rw + 15, c0[1] + rh + 4], fill=dark(gcol, 1.25) if False else mixc(gcol, (120, 90, 90), 0.3))
        d.ellipse([c0[0] - rw, c0[1] - rh, c0[0] + rw, c0[1] + rh], fill=(18, 14, 22))
    # crapaturi
    w = max(1, int(3 * cam.z))
    if 10.7 < t < 22.0:
        for ln in CRACK1:
            d.line([cam.p(*q) for q in ln[:1 + int(5 * seg(t, 10.7, 10.85))]], fill=BLACK, width=w)
    if 18.8 < t < 22.0:
        g = seg(t, 18.8, 21.5)
        for ln in CRACK2:
            d.line([cam.p(*q) for q in ln[:1 + int(5 * g)]], fill=BLACK, width=w)


# ----------------------------------------------------------------- efecte
def leaves(d, t, alpha=1.0):
    for i in range(28):
        r = random.Random(100 + i)
        sp = r.uniform(0.15, 0.4)
        x = (r.uniform(0, W + 200) - t * sp * W) % (W + 200) - 100
        y = r.uniform(60, 650) + math.sin(t * 2 + i) * 30
        a = t * r.uniform(3, 7) + i
        sz = r.uniform(5, 9)
        col = [(230, 120, 50), (200, 70, 60), (240, 170, 70), (120, 160, 70)][i % 4]
        pts = [(x + math.cos(a) * sz, y + math.sin(a) * sz * 0.5), (x - math.sin(a) * sz * 0.35, y + math.cos(a) * sz * 0.35),
               (x - math.cos(a) * sz, y - math.sin(a) * sz * 0.5), (x + math.sin(a) * sz * 0.35, y - math.cos(a) * sz * 0.35)]
        d.polygon(pts, fill=col + (int(220 * alpha),))


def debris(d, cam, t, t0, t1):
    k = seg(t, t0, t0 + 1.0) * (1 - seg(t, t1 - 0.3, t1))
    if k <= 0:
        return
    for i in range(30):
        r = random.Random(300 + i)
        x = r.uniform(80, 1200)
        sp = r.uniform(25, 70)
        y = GROUND - ((t - t0) * sp + r.uniform(0, 300)) % 320
        y += math.sin(t * 3 + i) * 4
        s = r.uniform(3, 9)
        a = t * 2 + i
        pts = [(x + math.cos(a + q * 2.1) * s, y + math.sin(a + q * 2.1) * s) for q in range(3)]
        d.polygon([cam.p(*q) for q in pts], fill=(40, 32, 40, int(230 * k)))


def speed_lines_h(d, t, fi, alpha=140):
    r = random.Random(fi)
    for _ in range(36):
        y = r.uniform(0, H)
        x = r.uniform(-200, W)
        ln = r.uniform(150, 500)
        d.line([(x, y), (x + ln, y)], fill=(255, 255, 255, alpha), width=r.choice([1, 2, 3]))


def focus_lines(d, fi, col=(255, 255, 255), n=70, inner=(260, 420), alpha=200):
    r = random.Random(fi // 2)
    cx, cy = W / 2, H / 2
    for _ in range(n):
        a = r.uniform(0, math.tau)
        ri = r.uniform(*inner)
        w = r.uniform(0.006, 0.02)
        pts = [(cx + math.cos(a) * ri, cy + math.sin(a) * ri),
               (cx + math.cos(a - w) * 1000, cy + math.sin(a - w) * 1000),
               (cx + math.cos(a + w) * 1000, cy + math.sin(a + w) * 1000)]
        d.polygon(pts, fill=col + (alpha,))


def starburst(d, c, r1, r2, n, rng, col, rot=0.0):
    pts = []
    for i in range(n * 2):
        a = rot + i * math.pi / n
        rr = rng.uniform(r2 * 0.6, r2) if i % 2 == 0 else r1
        pts.append((c[0] + math.cos(a) * rr, c[1] + math.sin(a) * rr))
    d.polygon(pts, fill=col)


def impacts(d, g, cam, t, fi):
    for (t0, x, y, s) in IMPACTS:
        a = t - t0
        if 0 <= a < 0.22:
            u = a / 0.22
            c = cam.p(x, y)
            z = cam.z * s
            rng = random.Random(int(t0 * 1000))
            starburst(g, c, 30 * z, 110 * z * (0.6 + u), 9, rng, (255, 200, 90), rng.uniform(0, 6))
            if u < 0.6:
                starburst(d, c, 14 * z, 80 * z * (0.5 + u), 9, rng, (255, 255, 230), rng.uniform(0, 6))
                circ(d, c, 14 * z, (255, 255, 255))
            rr = 30 * z + 140 * z * u
            d.ellipse([c[0] - rr, c[1] - rr * 0.8, c[0] + rr, c[1] + rr * 0.8], outline=(255, 255, 255),
                      width=max(1, int(6 * z * (1 - u))))


def dust_fx(o, cam, t):
    for (t0, x, y, s) in DUSTS:
        a = t - t0
        if 0 <= a < 1.0:
            r = random.Random(int(t0 * 100) + int(x))
            for i in range(14):
                side = -1 if i % 2 else 1
                dx = side * r.uniform(10, 120) * s * (0.3 + a)
                dy = -r.uniform(0, 45) * s * (0.3 + a)
                rad = r.uniform(14, 30) * s * (0.6 + a) * cam.z
                p = cam.p(x + dx, y + dy)
                circ(o, p, rad, (190, 160, 140, int(170 * (1 - a))))


def smoke_fx(o, cam, t):
    for (t0, x, y) in SMOKES:
        a = t - t0
        if 0 <= a < 0.6:
            r = random.Random(int(t0 * 100))
            for i in range(12):
                ang = r.uniform(0, math.tau)
                dist = r.uniform(10, 60) * (0.4 + a * 2)
                rad = r.uniform(25, 45) * (0.7 + a) * cam.z
                p = cam.p(x + math.cos(ang) * dist, y + math.sin(ang) * dist * 0.8)
                circ(o, p, rad, (235, 235, 240, int(230 * (1 - a / 0.6))))


def bolt(d, a, b, rng, col, w, segs=7, jit=14):
    pts = [a]
    for i in range(1, segs):
        u = i / segs
        pts.append((lerp(a[0], b[0], u) + rng.uniform(-jit, jit), lerp(a[1], b[1], u) + rng.uniform(-jit, jit)))
    pts.append(b)
    d.line(pts, fill=col, width=w)


def sphere(d, g, cam, c, R, t, fi):
    sc = cam.p(*c)
    r = R * cam.z
    circ(g, sc, r * 2.0, (30, 100, 255))
    circ(g, sc, r * 1.2, (130, 210, 255))
    circ(d, sc, r, (120, 195, 255))
    circ(d, sc, r * 0.78, (195, 235, 255))
    circ(d, sc, r * 0.45, (255, 255, 255))
    for k in range(6):
        rr = r * (0.55 + 0.08 * k)
        ang = (t * 1100 * (1 if k % 2 else -1) + k * 60) % 360
        d.arc([sc[0] - rr, sc[1] - rr, sc[0] + rr, sc[1] + rr], ang, ang + 110,
              fill=(255, 255, 255) if k % 2 else (60, 150, 255), width=max(2, int(3 * cam.z)))
    rng = random.Random(fi * 7 + 1)
    for _ in range(5):
        a = rng.uniform(0, math.tau)
        ln = r * rng.uniform(1.2, 2.2)
        e = (sc[0] + math.cos(a) * ln, sc[1] + math.sin(a) * ln)
        s0 = (sc[0] + math.cos(a) * r * 0.6, sc[1] + math.sin(a) * r * 0.6)
        bolt(g, s0, e, rng, (90, 180, 255), max(2, int(6 * cam.z)), jit=8 * cam.z)
        bolt(d, s0, e, rng, (240, 250, 255), max(1, int(2 * cam.z)), jit=8 * cam.z)


FIRE_COLS = [(200, 30, 20), (255, 90, 20), (255, 160, 40), (255, 230, 120), (255, 255, 220)]


def fireball(d, g, cam, c, R, t, fi):
    sc = cam.p(*c)
    r = R * cam.z
    circ(g, sc, r * 2.2, (255, 70, 0))
    rng = random.Random(fi * 13 + 5)
    for layer, col in enumerate(FIRE_COLS):
        k = 1 - layer * 0.2
        for _ in range(7 - layer):
            ox = rng.uniform(-r, r) * 0.45 * k
            oy = rng.uniform(-r, r) * 0.45 * k - r * 0.15
            circ(d, (sc[0] + ox, sc[1] + oy), r * k * rng.uniform(0.6, 0.8), col)
    for _ in range(5):
        a = rng.uniform(-2.6, -0.5)
        tip = (sc[0] + math.cos(a) * r * 1.8, sc[1] + math.sin(a) * r * 1.8)
        b1 = (sc[0] + math.cos(a + 0.5) * r * 0.6, sc[1] + math.sin(a + 0.5) * r * 0.6)
        b2 = (sc[0] + math.cos(a - 0.5) * r * 0.6, sc[1] + math.sin(a - 0.5) * r * 0.6)
        d.polygon([b1, tip, b2], fill=(255, 140, 30))


def fire_dragon(d, g, cam, start, front_x, t, fi, thick=1.0):
    """Raza de foc cu cap de dragon, de la `start` spre stanga pana la front_x."""
    sx, sy = start
    length = sx - front_x
    if length <= 4:
        return
    n = max(4, int(length / 9))
    rng = random.Random(fi * 3)
    pts = []
    for i in range(n + 1):
        u = i / n
        x = sx - length * u
        y = sy + math.sin(u * 9 - t * 18) * 14 * u - 10 * u
        pts.append((x, y, u))
    for li, col in enumerate(FIRE_COLS[:4]):
        k = 1 - li * 0.24
        for (x, y, u) in pts:
            rad = (14 + 26 * u) * thick * k * rng.uniform(0.85, 1.15)
            p = cam.p(x + rng.uniform(-4, 4), y + rng.uniform(-4, 4))
            if li == 0:
                circ(g, p, rad * cam.z * 1.6, (255, 60, 0))
            circ(d, p, rad * cam.z, col)
    # cap de dragon
    hx, hy, _ = pts[-1]
    hr = 46 * thick
    hp = cam.p(hx, hy)
    z = cam.z
    for (ax, ay, bx, by) in ((0.2, -0.8, 0.9, -1.6), (0.45, -0.6, 1.2, -1.2)):
        d.polygon([cam.p(hx + hr * ax, hy + hr * ay), cam.p(hx + hr * bx, hy + hr * by),
                   cam.p(hx + hr * (ax + 0.35), hy + hr * (ay + 0.15))], fill=(255, 120, 30))
    circ(g, hp, hr * z * 1.8, (255, 90, 0))
    circ(d, hp, hr * z, (255, 110, 30))
    circ(d, hp, hr * z * 0.75, (255, 180, 60))
    d.polygon([cam.p(hx - hr * 0.9, hy - hr * 0.1), cam.p(hx - hr * 0.2, hy - hr * 0.55),
               cam.p(hx - hr * 0.1, hy + hr * 0.5)], fill=(255, 220, 110))
    jaw = 0.25 + 0.15 * math.sin(t * 20)
    d.polygon([cam.p(hx - hr * 1.1, hy + hr * 0.05), cam.p(hx - hr * 0.2, hy + hr * jaw),
               cam.p(hx - hr * 0.3, hy + hr * 0.1)], fill=(120, 20, 10))
    e = cam.p(hx - hr * 0.35, hy - hr * 0.3)
    d.ellipse([e[0] - 7 * z, e[1] - 4 * z, e[0] + 7 * z, e[1] + 4 * z], fill=(255, 255, 230))
    d.ellipse([e[0] - 2 * z, e[1] - 4 * z, e[0] + 2 * z, e[1] + 4 * z], fill=(60, 0, 0))


def clash_flare(d, g, o, cam, c, t, fi, power):
    sc = cam.p(*c)
    z = cam.z
    rng = random.Random(fi * 11)
    starburst(g, sc, 70 * z * power, 260 * z * power, 12, rng, (200, 160, 255), rng.uniform(0, 6))
    starburst(d, sc, 30 * z * power, 170 * z * power, 10, rng, (255, 255, 240), rng.uniform(0, 6))
    circ(d, sc, 34 * z * power, (255, 255, 255))
    for i in range(40):
        r2 = random.Random(i * 31)
        life = 0.5
        a = ((t * 1.7 + r2.uniform(0, 1)) % 1.0)
        ang = r2.uniform(-math.pi, math.pi)
        sp = r2.uniform(250, 600)
        dist = a * sp * life * 2
        p1 = (c[0] + math.cos(ang) * dist, c[1] + math.sin(ang) * dist * 0.7 + a * a * 120)
        p0 = (c[0] + math.cos(ang) * (dist - 30), c[1] + math.sin(ang) * (dist - 30) * 0.7 + a * a * 120)
        col = (130, 200, 255) if i % 2 else (255, 170, 60)
        d.line([cam.p(*p0), cam.p(*p1)], fill=col, width=max(1, int(3 * z)))
    for _ in range(3):
        a = rng.uniform(0, math.tau)
        e = (sc[0] + math.cos(a) * 300 * z, sc[1] + math.sin(a) * 200 * z)
        bolt(g, sc, e, rng, (120, 200, 255), max(2, int(7 * z)), segs=9, jit=25 * z)
        bolt(d, sc, e, rng, (255, 255, 255), max(1, int(2 * z)), segs=9, jit=25 * z)


def explosion(o, g, cam, t):
    a = t - 22.0
    if a < 0 or a > 3.2:
        return
    c = cam.p(640, 430)
    z = cam.z
    # minge de foc
    fa = 1 - seg(a, 0.9, 1.8)
    if fa > 0:
        r = 1100 * (1 - math.exp(-a * 3.2)) * z
        circ(g, c, r * 1.1, (255, 120, 30))
        circ(o, c, r, (255, 120, 40, int(255 * fa)))
        circ(o, c, r * 0.78, (255, 200, 90, int(255 * fa)))
        circ(o, c, r * 0.55 * (1 - seg(a, 0, 1.4)), (255, 255, 235, int(255 * fa)))
    # fum
    sa = seg(a, 0.6, 1.1) * (1 - seg(a, 2.0, 3.2))
    if sa > 0:
        for i in range(40):
            r2 = random.Random(i * 17)
            ang = r2.uniform(0, math.tau)
            dist = r2.uniform(0, 700) * (0.6 + a * 0.25)
            p = (c[0] + math.cos(ang) * dist * z, c[1] + math.sin(ang) * dist * 0.45 * z - a * 20)
            rad = r2.uniform(120, 220) * (0.8 + a * 0.2) * z
            gcol = int(lerp(90, 60, r2.random()))
            circ(o, p, rad, (gcol, gcol - 8, gcol + 10, int(235 * sa)))
    # unda de soc
    if a < 0.8:
        u = a / 0.8
        r = 1600 * u * z
        o.ellipse([c[0] - r, c[1] - r * 0.55, c[0] + r, c[1] + r * 0.55], outline=(255, 255, 255, int(255 * (1 - u))),
                  width=max(2, int(30 * (1 - u))))


def embers(o, t, k):
    for i in range(40):
        r = random.Random(500 + i)
        x = (r.uniform(0, W) + math.sin(t + i) * 30) % W
        y = H - ((t * r.uniform(30, 80) + r.uniform(0, H)) % (H + 50))
        rad = r.uniform(1.5, 3.5)
        circ(o, (x, y), rad, (255, r.randint(120, 200), 60, int(220 * k)))


def text(d, xy, s, size, fill, stroke=6, sfill=BLACK, anchor="mm", heavy=False):
    d.text(xy, s, font=font(size, heavy), fill=fill, stroke_width=stroke, stroke_fill=sfill, anchor=anchor)


# ----------------------------------------------------------------- scena ochilor
def eye_panel(st, t, narrow, flip):
    pw, ph = W, 330
    img = Image.new("RGB", (pw, ph), st["skin"])
    d = ImageDraw.Draw(img, "RGBA")
    # umbra si linii de viteza
    for i in range(ph):
        if i > ph * 0.7:
            d.line([(0, i), (pw, i)], fill=dark(st["skin"], 1 - 0.15 * (i - ph * 0.7) / (ph * 0.3)))
    r = random.Random(int(t * 12))
    for _ in range(25):
        y = r.uniform(0, ph)
        x = r.uniform(-100, pw)
        d.line([(x, y), (x + r.uniform(200, 600), y)], fill=dark(st["skin"], 0.85) + (120,), width=2)
    cx, cy = pw / 2 + (60 if not flip else -60), ph * 0.58
    ew, eh = 230, 105 * narrow

    def upper(u):
        return cy - eh * (1 - u * u) * (1 + 0.25 * u * (-1 if flip else 1))

    def lower(u):
        return cy + eh * 0.55 * (1 - u * u)

    us = np.linspace(-1, 1, 40)
    up = [(cx + ew * u, upper(u)) for u in us]
    lo = [(cx + ew * u, lower(u)) for u in us[::-1]]
    mask = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(mask).polygon(up + lo, fill=255)
    eye = Image.new("RGB", (pw, ph), (250, 250, 252))
    ed = ImageDraw.Draw(eye)
    ic = (cx + (-15 if flip else 15), cy + 5)
    ir = 88
    base = st["eye"]
    for k in range(10, 0, -1):
        circ(ed, ic, ir * k / 10, mixc(dark(base, 0.35), base, 1 - k / 12))
    if st["kind"] == "spiky":
        rr = random.Random(3)
        for _ in range(14):
            a = rr.uniform(0, math.tau)
            ed.line([(ic[0] + math.cos(a) * 34, ic[1] + math.sin(a) * 34),
                     (ic[0] + math.cos(a) * 80, ic[1] + math.sin(a) * 80)], fill=(170, 225, 255), width=3)
        circ(ed, ic, 32, BLACK)
    else:
        ed.ellipse([ic[0] - 58, ic[1] - 58, ic[0] + 58, ic[1] + 58], outline=BLACK, width=6)
        ed.ellipse([ic[0] - 11, ic[1] - 44, ic[0] + 11, ic[1] + 44], fill=BLACK)
    circ(ed, (ic[0] - 30, ic[1] - 30), 15, (255, 255, 255))
    circ(ed, (ic[0] + 25, ic[1] + 22), 7, (255, 255, 255))
    shing = seg(t, 5.25, 5.35) * (1 - seg(t, 5.35, 5.6))
    if shing > 0:
        starburst(ed, (ic[0] - 30, ic[1] - 30), 6, 70 * shing, 4, random.Random(1), (255, 255, 255))
    img.paste(eye, (0, 0), mask)
    d.line(up, fill=BLACK, width=11)
    wing_x = cx + (ew if not flip else -ew)
    d.polygon([(wing_x, upper(1 if not flip else -1)), (wing_x + (40 if not flip else -40), cy - eh * 0.45),
               (wing_x - (30 if not flip else -30), cy - 8)], fill=BLACK)
    d.line(lo, fill=dark(st["skin"], 0.5), width=4)
    # spranceana
    inner = cx - ew * 0.9 if not flip else cx + ew * 0.9
    outer = cx + ew * 1.0 if not flip else cx - ew * 1.0
    by = cy - 105 - (1 - narrow) * 30
    d.polygon([(inner, by + 45 + (1 - narrow) * 40), (outer, by - 10), (outer, by + 12), (inner, by + 65 + (1 - narrow) * 40)], fill=BLACK)
    # suvite
    rr = random.Random(7 if flip else 8)
    for i in range(7):
        x = rr.uniform(0, pw)
        w = rr.uniform(60, 140)
        tip = rr.uniform(40, 110) + 6 * math.sin(t * 6 + i)
        d.polygon([(x - w, -5), (x + w, -5), (x + rr.uniform(-40, 40), tip)], fill=st["hair"], outline=BLACK, width=3)
    return img


def render_eyes(t, fi):
    img = Image.new("RGB", (W, H), (10, 8, 16))
    d = ImageDraw.Draw(img, "RGBA")
    focus_lines(d, fi, n=90, inner=(80, 200), alpha=170)
    narrow = 1 - 0.45 * ease(seg(t, 5.15, 5.3))
    shapes = [
        (KAITO, [(0, 40), (W, 40), (W, 330), (0, 365)], 40, False, seg(t, 4.0, 4.3), -1),
        (RYUGA, [(0, 385), (W, 350), (W, 680), (0, 680)], 350, True, seg(t, 4.25, 4.55), 1),
    ]
    for st, poly, y0, flip, sl, sd in shapes:
        if sl <= 0:
            continue
        off = int(sd * W * (1 - ease(sl)))
        panel = eye_panel(st, t, narrow, flip)
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).polygon([(x + off, y) for x, y in poly], fill=255)
        layer = Image.new("RGB", (W, H))
        layer.paste(panel, (off, y0))
        img.paste(layer, (0, 0), mask)
        d.line([(x + off, y) for x, y in poly] + [(poly[0][0] + off, poly[0][1])], fill=(255, 255, 255), width=6)
    a = seg(t, 4.6, 4.9)
    if a > 0:
        text(d, (180, 300), "KAITO", 44, (120, 220, 255) + (int(255 * a),), stroke=5)
        text(d, (W - 180, 640), "RYUGA", 44, (255, 90, 90) + (int(255 * a),), stroke=5)
    return img


# ----------------------------------------------------------------- frame
def render(fi):
    t = fi / FPS
    if 4.0 <= t < 6.0:
        img = render_eyes(t, fi)
        return finish(img, t, fi)
    cam = camera(t)
    img = Image.new("RGB", (W, H))
    draw_background(img, cam, t)
    back = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(back)
    glow = Image.new("RGB", (W, H))
    g = ImageDraw.Draw(glow)

    if t < 9 or t > 24.5:
        leaves(bd, t, 1.0 if t < 9 else seg(t, 24.5, 26))
    debris(bd, cam, t, 14.5, 22.0)
    dust_fx(bd, cam, t)
    if any(a <= t < b for a, b in AFTER):
        for who, keys, st in (("K", K_KEYS, KAITO), ("R", R_KEYS, RYUGA)):
            if hidden(who, t):
                continue
            for k, dt in enumerate((0.1, 0.06, 0.03)):
                x, air, p, f = state(keys, t - dt)
                col = st["aura"] + (60 + 40 * k,)
                draw_fighter(bd, cam, x, air, p, f, st, t, tint=col)
    if 6.2 < t < 7.2 or 18.1 < t < 18.8:
        speed_lines_h(bd, t, fi)
    img.paste(back, (0, 0), back)
    d = ImageDraw.Draw(img)

    # aure in timpul incarcarii
    states = {}
    for who, keys, st in (("K", K_KEYS, KAITO), ("R", R_KEYS, RYUGA)):
        states[who] = state(keys, t)
        if 14.3 < t < 22.0 and not hidden(who, t):
            x, air, p, f = states[who]
            j = joints_world(x, air, p, f)
            k = seg(t, 14.3, 15.3)
            hip = cam.p(*j["hip"])
            rr = random.Random(fi + (0 if who == "K" else 99))
            for _ in range(10):
                ox = rr.uniform(-60, 60) * cam.z
                h = rr.uniform(120, 260) * cam.z * k
                bx = hip[0] + ox
                by = hip[1] + 90 * cam.z
                g.polygon([(bx - 20 * cam.z, by), (bx + 20 * cam.z, by), (bx + rr.uniform(-15, 15) * cam.z, by - h)],
                          fill=dark(st["aura"], 0.8))

    joints = {}
    for who, st in (("R", RYUGA), ("K", KAITO)):
        if hidden(who, t):
            continue
        x, air, p, f = states[who]
        if 18.8 < t < 22.0:
            x += math.sin(t * 60 + (0 if who == "K" else 1)) * 2
        joints[who] = draw_fighter(d, cam, x, air, p, f, st, t)

    front = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    fd = ImageDraw.Draw(front)

    # tehnici
    if "K" in joints and 15.2 <= t < 22.0:
        hf = joints["K"]["hf"]
        R = 45 * ease(seg(t, 15.2, 16.0)) * (1 + 0.15 * seg(t, 18.8, 21.9)) + 3 * math.sin(t * 40)
        sc = (hf[0] + R + 2, hf[1])
        sphere(d, g, cam, sc, max(2, R), t, fi)
        joints["sphere"] = (sc, R)
    if "K" in joints and 14.3 < t < 15.0 and int(t * 8) % 2 == 0:
        hf = cam.p(*joints["K"]["hf"])
        starburst(g, hf, 6, 40 * cam.z, 6, random.Random(fi), (120, 200, 255))
    if "R" in joints and 14.6 < t < 15.6 and int(t * 8) % 2 == 0:
        hf = cam.p(*joints["R"]["hf"])
        starburst(g, hf, 6, 40 * cam.z, 6, random.Random(fi), (255, 140, 60))
    if "R" in joints and 15.6 <= t < 22.0:
        hf = joints["R"]["hf"]
        if t < 18.0:
            R = 30 * ease(seg(t, 15.6, 16.6)) + 3 * math.sin(t * 30)
            fireball(d, g, cam, (hf[0] - 20, hf[1] - 5), max(2, R), t, fi)
        else:
            start = (hf[0] - 10, hf[1])
            target = joints["sphere"][0][0] + joints["sphere"][1] + 20 if "sphere" in joints else 700
            fx = lerp(start[0], target, ease(seg(t, 18.0, 18.8)))
            if t > 18.8:
                fx = target + 6 * math.sin(t * 25)
            fire_dragon(d, g, cam, start, fx, t, fi, thick=1.0 + 0.3 * seg(t, 18.8, 21.9))
            fireball(d, g, cam, start, 30, t, fi)
    if t >= 18.8 and t < 22.0 and "sphere" in joints:
        sc, R = joints["sphere"]
        cp = (sc[0] + R + 10, sc[1])
        clash_flare(d, g, fd, cam, cp, t, fi, 0.7 + 0.5 * seg(t, 18.8, 21.9) + 0.1 * math.sin(t * 30))

    impacts(d, g, cam, t, fi)
    smoke_fx(fd, cam, t)
    explosion(fd, g, cam, t)
    if t > 24.5:
        embers(fd, t, seg(t, 24.5, 25.5))
    # linii de focalizare
    if 7.2 <= t < 7.6 or 10.3 <= t < 10.5 or 11.4 <= t < 11.6 or 15.15 <= t < 17.3 or 18.8 <= t < 21.95:
        fl_col = (255, 255, 255) if t < 15 or t > 18 else (10, 10, 20)
        focus_lines(fd, fi, col=fl_col, n=60, inner=(330, 460), alpha=170)
    img.paste(front, (0, 0), front)

    # stralucire
    gs = glow.resize((W // 2, H // 2), Image.BILINEAR).filter(ImageFilter.GaussianBlur(14)).resize((W, H), Image.BILINEAR)
    img = ImageChops.add(img, gs)
    img = ImageChops.add(img, glow.filter(ImageFilter.GaussianBlur(2)), scale=2.5)

    # blitz alb la explozie
    fa = 1 - seg(t, 22.1, 22.7) if t >= 22.0 else 0
    if fa > 0:
        img = Image.blend(img, Image.new("RGB", (W, H), (255, 255, 255)), fa)
    return finish(img, t, fi)


def finish(img, t, fi):
    # cadre de impact inversate (alb-negru)
    if any(a <= t < b for a, b in INVERT):
        gray = img.convert("L").point(lambda v: 255 if v < 110 else 0)
        img = gray.convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    # titluri
    if t < 4.0:
        out = 1 - seg(t, 3.6, 4.0)
        a1 = ease(seg(t, 1.0, 1.3))
        if a1 > 0:
            text(d, (330 - 400 * (1 - a1), 610), "KAITO", 62, (120, 220, 255, int(255 * out)), stroke=7)
            text(d, (330 - 400 * (1 - a1), 655), "Stilul Furtunii", 24, (255, 255, 255, int(255 * out)), stroke=4)
        a2 = ease(seg(t, 1.6, 1.9))
        if a2 > 0:
            text(d, (950 + 400 * (1 - a2), 610), "RYUGA", 62, (255, 90, 90, int(255 * out)), stroke=7)
            text(d, (950 + 400 * (1 - a2), 655), "Arta Flăcării", 24, (255, 255, 255, int(255 * out)), stroke=4)
        a3 = seg(t, 2.4, 2.5)
        if a3 > 0:
            sz = int(150 + 120 * (1 - ease(seg(t, 2.4, 2.65))))
            text(d, (640, 300), "VS", sz, (255, 210, 60, int(255 * out)), stroke=10, heavy=True)
    # numele tehnicilor
    for t0, t1, s, col in ((15.25, 16.25, "Stilul Furtunii: SFERA TUNETULUI!", (120, 220, 255)),
                           (16.35, 17.35, "Arta Flăcării: DRAGONUL STACOJIU!", (255, 120, 80))):
        if t0 <= t < t1:
            a = seg(t, t0, t0 + 0.12) * (1 - seg(t, t1 - 0.12, t1))
            d.rectangle([0, 600, W, 662], fill=(0, 0, 0, int(150 * a)))
            text(d, (640, 631), s, 40, col + (int(255 * a),), stroke=5)
    # final
    if t > 27.2:
        a = seg(t, 27.2, 27.8)
        text(d, (640, 330), "CONTINUAREA", 70, (255, 255, 255, int(255 * a)), stroke=8, heavy=True)
        text(d, (640, 400), "în episodul următor...", 34, (255, 210, 60, int(255 * a)), stroke=5)
    # fade
    fade = 0.0
    if t < 0.6:
        fade = 1 - t / 0.6
    if t > 29.2:
        fade = seg(t, 29.2, 30.0)
    if 3.9 <= t < 4.0 or 5.9 <= t < 6.0:
        fade = 0.6
    if fade > 0:
        d.rectangle([0, 0, W, H], fill=(0, 0, 0, int(255 * fade)))
    # benzi cinematice
    d.rectangle([0, 0, W, 34], fill=(0, 0, 0))
    d.rectangle([0, H - 34, W, H], fill=(0, 0, 0))
    return img


# ----------------------------------------------------------------- audio
SR = 44100


def build_audio(path):
    n_total = int(SR * (DUR + 1))
    L = np.zeros(n_total)
    Rr = np.zeros(n_total)
    rng = np.random.default_rng(1)

    def add(t, sig, gain=1.0, pan=0.0):
        i = int(t * SR)
        if i >= n_total:
            return
        sig = sig[: n_total - i] * gain
        L[i:i + len(sig)] += sig * math.sqrt((1 - pan) / 2)
        Rr[i:i + len(sig)] += sig * math.sqrt((1 + pan) / 2)

    def band(dur, lo, hi):
        n = int(dur * SR)
        X = np.fft.rfft(rng.standard_normal(n))
        fr = np.fft.rfftfreq(n, 1 / SR)
        X[(fr < lo) | (fr > hi)] = 0
        y = np.fft.irfft(X, n)
        return y / (np.abs(y).max() + 1e-9)

    def tt(dur):
        return np.arange(int(dur * SR)) / SR

    def tone(f0, f1, dur, decay):
        x = tt(dur)
        ph = 2 * np.pi * np.cumsum(np.linspace(f0, f1, len(x))) / SR
        return np.sin(ph) * np.exp(-x * decay)

    def hit(big=1.0):
        d = 0.18 + 0.25 * (big - 0.6)
        x = tt(d)
        s = band(d, 150, 5000) * np.exp(-x * 30 / big) * 0.8
        s += tone(110, 45, d, 12 / big) * 1.0
        return s

    def whoosh(d=0.35):
        x = tt(d)
        return band(d, 500, 3500) * np.sin(np.pi * x / d) ** 2 * 0.5

    def taiko(acc=1.0):
        x = tt(0.6)
        return (tone(120, 55, 0.6, 7) + band(0.6, 100, 900) * np.exp(-x * 40) * 0.4) * acc

    def shing():
        x = tt(1.2)
        s = sum(np.sin(2 * np.pi * f * x) * np.exp(-x * dcy) for f, dcy in ((2300, 3), (3450, 4), (4720, 5), (6100, 7)))
        return s * 0.25 * np.minimum(1, x * 200)

    def charge(dur, f0, f1, kind):
        x = tt(dur)
        fr = np.linspace(f0, f1, len(x)) * (1 + 0.03 * np.sin(2 * np.pi * 7 * x))
        ph = 2 * np.pi * np.cumsum(fr) / SR
        saw = 2 * ((ph / (2 * np.pi)) % 1) - 1
        env = np.minimum(1, x / (dur * 0.6))
        if kind == "bolt":
            s = (0.35 * saw + 0.5 * np.sin(ph)) * env
            cr = band(dur, 2000, 9000) * (rng.random(len(x)) < 0.004).astype(float)
            cr = np.convolve(cr, np.exp(-np.arange(400) / 60), mode="same")
            s += cr * 3 * env
        else:
            s = band(dur, 60, 1500) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * x)) * env + 0.2 * np.sin(ph) * env
        return s

    def explosion():
        d = 4.0
        x = tt(d)
        s = band(d, 25, 2500) * np.exp(-x * 1.3)
        s += tone(55, 28, d, 1.0) * 1.2
        s += band(d, 2000, 8000) * np.exp(-x * 4) * 0.3
        return s

    def flute(t0, notes, beat):
        tcur = t0
        for nt, ln in notes:
            if nt is not None:
                f = 440 * 2 ** ((nt - 69) / 12)
                d = ln * beat
                x = tt(d + 0.3)
                vib = 1 + 0.008 * np.sin(2 * np.pi * 5.5 * x) * np.minimum(1, x * 3)
                ph = 2 * np.pi * np.cumsum(f * vib) / SR
                env = np.minimum(1, x / 0.06) * np.exp(-np.maximum(0, x - d) * 12)
                s = (np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)) * env
                s += band(d + 0.3, 1500, 6000) * env * 0.04
                add(tcur, s, 0.22)
            tcur += ln * beat

    # dron de fundal
    x = tt(DUR)
    tense = np.clip((x - 5.5) / 1.0, 0, 1) * (1 - np.clip((x - 22) / 0.5, 0, 1))
    drone = (np.sin(2 * np.pi * 73.4 * x) + 0.6 * np.sin(2 * np.pi * 110 * x) + 0.4 * np.sin(2 * np.pi * 146.8 * x))
    drone *= (0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * x)) * (0.05 + 0.05 * tense)
    drone *= np.minimum(1, x / 1.0) * (1 - np.clip((x - 28.5) / 1.5, 0, 1))
    add(0, drone)
    # melodie (intro si final) - pentatonica re minor
    D, F, G, A, C, D2 = 74, 77, 79, 81, 84, 86
    flute(0.3, [(D, 1), (F, 0.5), (G, 0.5), (A, 1.5), (None, 0.5), (C, 0.5), (A, 0.5), (G, 1), (A, 2)], 0.42)
    flute(25.2, [(A, 1), (G, 0.5), (F, 0.5), (D, 1.5), (None, 0.5), (F, 0.5), (G, 0.5), (D, 3)], 0.5)
    # tobe taiko
    beat = 60 / 140
    tb = 6.0
    i = 0
    while tb < 21.9:
        if not (14.0 <= tb < 15.1):
            acc = 1.0 if i % 4 == 0 else 0.55
            add(tb, taiko(acc), 0.6)
            if i % 4 == 3:
                add(tb + beat / 2, taiko(0.4), 0.6)
        i += 1
        tb += beat
    for k in range(8):
        add(17.4 + k * 0.075, taiko(0.6 + k * 0.05), 0.6)
    add(2.4, taiko(1.3), 0.9)
    add(2.4, hit(1.4), 0.6)
    add(4.0, whoosh(), 0.8, -0.5)
    add(4.25, whoosh(), 0.8, 0.5)
    add(5.22, shing(), 0.9)
    add(6.1, whoosh(0.5), 0.7, -0.4)
    add(6.15, whoosh(0.5), 0.7, 0.4)
    for (t0, xx, yy, s) in IMPACTS:
        add(t0, hit(max(0.6, s)), 0.75 if s >= 1 else 0.45, (xx - 640) / 900)
        if s >= 1:
            add(t0 - 0.12, whoosh(0.2), 0.4, (xx - 640) / 900)
    for (t0, xx, yy) in SMOKES:
        add(t0, band(0.3, 300, 2500) * np.exp(-tt(0.3) * 10), 0.4)
    for t0 in (9.65, 13.65):
        add(t0, whoosh(0.4), 0.6)
    add(14.2, charge(1.0, 400, 900, "bolt"), 0.2, -0.6)
    add(15.2, charge(6.7, 90, 260, "bolt"), 0.35, -0.5)
    add(15.6, charge(6.3, 60, 120, "fire"), 0.35, 0.5)
    add(18.0, whoosh(0.8), 0.7)
    add(18.8, hit(2.0), 0.9)
    add(18.8, band(3.2, 80, 6000) * (0.6 + 0.4 * np.sin(2 * np.pi * 9 * tt(3.2))), 0.25)
    add(22.0, explosion(), 1.0)
    mix = np.stack([L, Rr], axis=1)
    mix = np.tanh(mix * 1.3)
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.9
    data = (mix[: int(SR * DUR)] * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def _frame_bytes(fi):
    return render(fi).tobytes()


def main():
    out = os.path.join(HERE, "kaito_vs_ryuga.mp4")
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        for t in [float(a) for a in sys.argv[2:]]:
            render(int(t * FPS)).save(os.path.join(HERE, f"preview_{t:05.2f}.png"))
        return
    wav = os.path.join(HERE, "_audio.wav")
    build_audio(wav)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "19",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool() as pool:
        for i, b in enumerate(pool.imap(_frame_bytes, range(N), chunksize=4)):
            proc.stdin.write(b)
            if i % 48 == 0:
                print(f"cadru {i}/{N}", flush=True)
    proc.stdin.close()
    proc.wait()
    os.remove(wav)
    print("Gata:", out)


if __name__ == "__main__":
    main()
