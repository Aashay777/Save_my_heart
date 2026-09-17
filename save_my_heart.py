import math
import random
import sys
import numpy as np
import pygame

pygame.init()
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2)
    SOUND_AVAILABLE = True
except pygame.error:
    SOUND_AVAILABLE = False

INFO = pygame.display.Info()
W, H = 1000, 700
screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
pygame.display.set_caption("Save my heart")
clock = pygame.time.Clock()

PAPER = (242, 222, 214)
PAPER_DEEP = (224, 191, 180)
INK = (43, 27, 42)
INK_SOFT = (107, 85, 102)
CARMINE = (214, 34, 74)
PLUM = (138, 92, 126)
WHITE_PINK = (255, 241, 236)

def load_font(names, size, bold=False):
    for name in names:
        try:
            f = pygame.font.SysFont(name, size, bold=bold)
            if f:
                return f
        except Exception:
            continue
    return pygame.font.SysFont(None, size, bold=bold)


SERIF_NAMES = ["Fraunces", "Georgia", "Times New Roman", "DejaVu Serif", "serif"]
SANS_NAMES = ["DM Sans", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"]


def serif(size, bold=True):
    return load_font(SERIF_NAMES, size, bold=bold)


def sans(size, bold=False):
    return load_font(SANS_NAMES, size, bold=bold)

def cubic_bezier(p0, p1, p2, p3, steps=22):
    pts = []
    for i in range(steps + 1):
        t = i / steps
        mt = 1 - t
        x = mt ** 3 * p0[0] + 3 * mt ** 2 * t * p1[0] + 3 * mt * t ** 2 * p2[0] + t ** 3 * p3[0]
        y = mt ** 3 * p0[1] + 3 * mt ** 2 * t * p1[1] + 3 * mt * t ** 2 * p2[1] + t ** 3 * p3[1]
        pts.append((x, y))
    return pts

_HEART_LOCAL = (
    cubic_bezier((0, -0.32), (0.52, -1.14), (1.42, -0.10), (0, 0.88))
    + cubic_bezier((0, 0.88), (-1.42, -0.10), (-0.52, -1.14), (0, -0.32))[1:]
)


def heart_points(cx, cy, scale):
    return [(cx + x * scale, cy + y * scale) for x, y in _HEART_LOCAL]


def rotate_translate(local_x, local_y, angle, tx, ty):
    ca, sa = math.cos(angle), math.sin(angle)
    return (local_x * ca - local_y * sa + tx, local_x * sa + local_y * ca + ty)


def draw_dashed_line(surface, color, p1, p2, width, dash_len, gap_len, offset=0.0, alpha=255):
    x1, y1 = p1
    x2, y2 = p2
    total_len = math.hypot(x2 - x1, y2 - y1)
    if total_len < 1:
        return
    dx, dy = (x2 - x1) / total_len, (y2 - y1) / total_len
    period = dash_len + gap_len
    pos = -offset % period
    if pos > 0:
        pos -= period
    line_surf = None
    if alpha < 255:
        line_surf = surface
    while pos < total_len:
        seg_start = max(pos, 0)
        seg_end = min(pos + dash_len, total_len)
        if seg_end > seg_start:
            sx, sy = x1 + dx * seg_start, y1 + dy * seg_start
            ex, ey = x1 + dx * seg_end, y1 + dy * seg_end
            pygame.draw.line(surface, color, (sx, sy), (ex, ey), width)
        pos += period


def with_alpha(color, alpha):
    return (color[0], color[1], color[2], max(0, min(255, int(alpha))))


def draw_circle_alpha(surface, color, center, radius, width=0):
    if radius <= 0:
        return
    tmp = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
    pygame.draw.circle(tmp, color, (radius + 2, radius + 2), radius, width)
    surface.blit(tmp, (center[0] - radius - 2, center[1] - radius - 2))


def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if font.size(test)[0] <= max_width or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

SAMPLE_RATE = 44100


def _tone(f0, f1, dur, peak=0.25, wave="sine"):
    n = max(1, int(SAMPLE_RATE * dur))
    t = np.linspace(0, dur, n, endpoint=False)
    freq = np.linspace(f0, max(20, f1), n)
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    if wave == "sine":
        w = np.sin(phase)
    elif wave == "triangle":
        frac = (phase / (2 * np.pi)) % 1.0
        w = 2 * np.abs(2 * (frac - np.floor(frac + 0.5))) - 1
    else:
        w = np.sin(phase)
    env = np.exp(-4.2 * t / max(dur, 1e-4))
    data = w * env * peak
    return data


def _noise(dur, f_center=1500, peak=0.18):
    n = max(1, int(SAMPLE_RATE * dur))
    raw = np.random.uniform(-1, 1, n)
    kernel = max(1, int(SAMPLE_RATE / max(200, f_center)))
    if kernel > 1:
        kernel_win = np.ones(kernel) / kernel
        smoothed = np.convolve(raw, kernel_win, mode="same")
        shaped = raw - smoothed * 0.6
    else:
        shaped = raw
    t = np.linspace(0, dur, n, endpoint=False)
    env = np.exp(-5.0 * t / max(dur, 1e-4))
    return shaped * env * peak


def _make_sound(mono):
    mono = np.clip(mono, -1, 1)
    data16 = (mono * 32767).astype(np.int16)
    stereo = np.column_stack([data16, data16]).copy(order="C")
    return pygame.sndarray.make_sound(stereo)


def _mix(*layers):
    n = max(len(l) for l in layers)
    out = np.zeros(n)
    for l in layers:
        out[: len(l)] += l
    return out


SOUNDS = {}
if SOUND_AVAILABLE:
    try:
        SOUNDS["arrow"] = _make_sound(_mix(_noise(0.22, 1400, 0.22), _tone(340, 96, 0.15, 0.14, "triangle")))
        SOUNDS["graze"] = _make_sound(_noise(0.18, 1800, 0.10))
        SOUNDS["break"] = _make_sound(_mix(_tone(190, 42, 0.6, 0.28), _tone(300, 70, 0.4, 0.12, "triangle"), _noise(0.45, 900, 0.16)))
        SOUNDS["start"] = _make_sound(
            _mix(
                _tone(523.25, 523.25, 0.34, 0.14),
                np.concatenate([np.zeros(int(SAMPLE_RATE * 0.06)), _tone(659.25, 659.25, 0.34, 0.14)]),
                np.concatenate([np.zeros(int(SAMPLE_RATE * 0.12)), _tone(987.77, 987.77, 0.34, 0.14)]),
            )
        )
        SOUNDS["ticks"] = [
            _make_sound(_tone(420 + i * 110, 470 + i * 130, 0.09, 0.11)) for i in range(3)
        ]
        SOUNDS["dodge"] = [
            _make_sound(
                _mix(
                    _tone(588 * (2 ** (n / 12)) * 0.72, 588 * (2 ** (n / 12)), 0.28, 0.17),
                    _tone(588 * (2 ** (n / 12)) * 1.5, 588 * (2 ** (n / 12)) * 2, 0.18, 0.06),
                )
            )
            for n in range(5)
        ]
    except Exception:
        SOUND_AVAILABLE = False


def play_panned(sound, x, width, volume=1.0):
    if not SOUND_AVAILABLE or sound is None or muted:
        return
    pan = max(-0.8, min(0.8, (x / max(1, width) - 0.5) * 1.6))
    left = min(1.0, 1.0 - pan) * volume
    right = min(1.0, 1.0 + pan) * volume
    ch = pygame.mixer.find_channel(True)
    if ch:
        ch.set_volume(max(0, left), max(0, right))
        ch.play(sound)


muted = False
_last_sfx_time = {}


def gated(name, gap, now):
    t = _last_sfx_time.get(name, -999)
    if now - t < gap:
        return False
    _last_sfx_time[name] = now
    return True


def sfx_arrow(px, now):
    if gated("arrow", 0.045, now):
        play_panned(SOUNDS.get("arrow"), px, W, 0.5)


def sfx_graze(px, now):
    if gated("graze", 0.09, now):
        play_panned(SOUNDS.get("graze"), px, W, 0.5)


def sfx_dodge(px, n):
    play_panned(SOUNDS["dodge"][n % 5], px, W, 0.55)


def sfx_break(px):
    play_panned(SOUNDS.get("break"), px, W, 0.7)


def sfx_tick(px, i):
    play_panned(SOUNDS["ticks"][min(i, 2)], px, W, 0.5)


def sfx_start(px):
    play_panned(SOUNDS.get("start"), px, W, 0.6)


TITLE, PLAYING, OVER = "title", "playing", "over"
mode = TITLE

heart = {"x": W / 2, "y": H / 2, "tx": W / 2, "ty": H / 2, "r": 15, "beat": 0.0, "alive": True}
arrows = []
bits = []
rings = []
score = 0
best = 0
broken = 0
level = 1
spawn_timer = 0.0
shake = 0.0
score_pop = 0.0
field_tint = 0.0

PAD_TIME = 0.7
pad = {"x": 0.0, "y": 0.0, "r": 48.0, "hold": 0.0, "phase": 0.0, "tick": 0, "inside": False}


def layout_pad():
    pad["r"] = max(40.0, min(62.0, min(W, H) * 0.09))
    if W < 820:
        pad["x"] = W * 0.5
        pad["y"] = H - max(118.0, H * 0.16)
    else:
        pad["x"] = W * 0.74
        pad["y"] = H * 0.55


layout_pad()

DIAG = math.hypot(W, H)


def rand(a, b):
    return a + random.random() * (b - a)


def difficulty():
    global level
    level = 1 + score // 5
    warn = max(0.34, 1.15 - 0.07 * (level - 1))
    speed = min(1150, 400 + 46 * (level - 1))
    every = max(0.36, 1.35 - 0.09 * (level - 1))
    two = min(0.45, 0.12 * (level - 3)) if level >= 4 else 0
    three = min(0.3, 0.07 * (level - 7)) if level >= 8 else 0
    return {"warn": warn, "speed": speed, "every": every, "two": two, "three": three}


def spawn_arrow(d):
    side = random.randint(0, 3)
    if side == 0:
        x, y = rand(-40, W + 40), -36
    elif side == 1:
        x, y = W + 36, rand(-40, H + 40)
    elif side == 2:
        x, y = rand(-40, W + 40), H + 36
    else:
        x, y = -36, rand(-40, H + 40)

    tx = heart["x"] + rand(-78, 78)
    ty = heart["y"] + rand(-78, 78)
    a = math.atan2(ty - y, tx - x)

    arrows.append(
        {
            "x": x, "y": y, "ox": x, "oy": y,
            "dx": math.cos(a), "dy": math.sin(a), "a": a,
            "state": "warn", "t": 0.0, "warn": d["warn"], "speed": d["speed"],
            "grazed": False,
        }
    )


def start_game():
    global mode, arrows, bits, rings, score, level, spawn_timer, shake, score_pop, field_tint
    mode = PLAYING
    arrows, bits, rings = [], [], []
    score, level = 0, 1
    spawn_timer, shake, score_pop, field_tint = 1.15, 0.0, 0.0, 0.0
    heart["alive"] = True
    pad["hold"], pad["tick"], pad["inside"] = 0.0, 0, False
    sfx_start(heart["x"])


def end_game():
    global mode, broken, best, shake, field_tint
    mode = OVER
    heart["alive"] = False
    broken += 1
    best = max(best, score)
    shake, field_tint = 18.0, 1.0
    sfx_break(heart["x"])
    for _ in range(26):
        a = rand(0, math.pi * 2)
        s = rand(90, 420)
        bits.append(
            {
                "x": heart["x"], "y": heart["y"],
                "vx": math.cos(a) * s, "vy": math.sin(a) * s - 60,
                "rot": rand(0, 6.28), "vr": rand(-9, 9), "life": 1.0, "size": rand(3, 9),
            }
        )

_bg_cache = {"surface": None, "size": None}


def make_grain_tile():
    tile = pygame.Surface((140, 140), pygame.SRCALPHA)
    arr = np.random.randint(-23, 24, (140, 140)).astype(np.int16)
    base = 128
    r = np.clip(base + arr, 0, 255).astype(np.uint8)
    g = np.clip(base + arr - 4, 0, 255).astype(np.uint8)
    b = np.clip(base + arr - 2, 0, 255).astype(np.uint8)
    a = np.full((140, 140), 16, dtype=np.uint8)
    px = pygame.surfarray.pixels3d(tile)
    px[..., 0] = r.T
    px[..., 1] = g.T
    px[..., 2] = b.T
    del px
    pa = pygame.surfarray.pixels_alpha(tile)
    pa[...] = a.T
    del pa
    return tile


GRAIN_TILE = make_grain_tile()


def build_background():
    surf = pygame.Surface((W, H))
    surf.fill(PAPER)
    for ty in range(0, H, 140):
        for tx in range(0, W, 140):
            surf.blit(GRAIN_TILE, (tx, ty))
    vignette = pygame.Surface((W, H), pygame.SRCALPHA)
    cx, cy = W / 2, H * 0.5
    max_r = DIAG * 0.62
    steps = 40
    for i in range(steps, 0, -1):
        t = i / steps
        r = int(max_r * t)
        alpha = int(0.42 * 255 * t)
        pygame.draw.circle(vignette, (*PLUM, alpha), (int(cx), int(cy)), r)
    surf.blit(vignette, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
    tint_base = pygame.Surface((W, H))
    tint_base.fill(PAPER_DEEP)
    tint_base.set_alpha(90)
    surf.blit(tint_base, (0, 0))
    _bg_cache["surface"] = surf
    _bg_cache["size"] = (W, H)


build_background()


def resize(new_w, new_h):
    global W, H, DIAG, screen
    W, H = max(320, new_w), max(320, new_h)
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    DIAG = math.hypot(W, H)
    layout_pad()
    build_background()

def draw_field():
    if _bg_cache["size"] != (W, H):
        build_background()
    screen.blit(_bg_cache["surface"], (0, 0))
    if field_tint > 0.001:
        tint = pygame.Surface((W, H), pygame.SRCALPHA)
        tint.fill(with_alpha(CARMINE, field_tint * 0.3 * 255))
        screen.blit(tint, (0, 0))


def draw_heart_shape():
    if not heart["alive"]:
        return
    beat = 1 + math.sin(heart["beat"]) * 0.055
    s = heart["r"] * beat

    shadow_pts = heart_points(heart["x"] + 3, heart["y"] + 7, s * 1.02)
    shadow_surf = pygame.Surface((W, H), pygame.SRCALPHA)
    pygame.draw.polygon(shadow_surf, with_alpha(INK, 0.16 * 255), shadow_pts)
    screen.blit(shadow_surf, (0, 0))

    main_pts = heart_points(heart["x"], heart["y"], s)
    pygame.draw.polygon(screen, CARMINE, main_pts)

    hi_pts = heart_points(heart["x"] - s * 0.38, heart["y"] - s * 0.34, s * 0.24)
    hi_surf = pygame.Surface((W, H), pygame.SRCALPHA)
    pygame.draw.polygon(hi_surf, with_alpha(WHITE_PINK, 0.35 * 255), hi_pts)
    screen.blit(hi_surf, (0, 0))


def draw_arrow(ar):
    length = 30
    tx = ar["x"] - ar["dx"] * length
    ty = ar["y"] - ar["dy"] * length
    px, py = -ar["dy"], ar["dx"]

    feather_surf = pygame.Surface((W, H), pygame.SRCALPHA)
    pygame.draw.line(
        feather_surf, with_alpha(PLUM, 0.22 * 255),
        (tx, ty), (tx - ar["dx"] * 46, ty - ar["dy"] * 46), 2,
    )
    screen.blit(feather_surf, (0, 0))

    pygame.draw.line(screen, INK, (tx, ty), (ar["x"] - ar["dx"] * 7, ar["y"] - ar["dy"] * 7), 3)

    head = [
        (ar["x"], ar["y"]),
        (ar["x"] - ar["dx"] * 13 + px * 6.5, ar["y"] - ar["dy"] * 13 + py * 6.5),
        (ar["x"] - ar["dx"] * 13 - px * 6.5, ar["y"] - ar["dy"] * 13 - py * 6.5),
    ]
    pygame.draw.polygon(screen, INK, head)

    fletch_surf = pygame.Surface((W, H), pygame.SRCALPHA)
    for i in range(2):
        bx, by = tx + ar["dx"] * (i * 7), ty + ar["dy"] * (i * 7)
        pygame.draw.line(fletch_surf, PLUM, (bx + px * 5.5, by + py * 5.5), (bx - ar["dx"] * 7 - px * 1.5, by - ar["dy"] * 7 - py * 1.5), 2)
        pygame.draw.line(fletch_surf, PLUM, (bx - px * 5.5, by - py * 5.5), (bx - ar["dx"] * 7 + px * 1.5, by - ar["dy"] * 7 + py * 1.5), 2)
    screen.blit(fletch_surf, (0, 0))


def draw_warning(ar):
    p = ar["t"] / ar["warn"]
    ex = ar["ox"] + ar["dx"] * DIAG * 1.4
    ey = ar["oy"] + ar["dy"] * DIAG * 1.4
    hot = p > 0.72

    color = CARMINE if hot else PLUM
    alpha = (0.55 + math.sin(ar["t"] * 46) * 0.2) if hot else (0.20 + p * 0.34)
    width = 3 if hot else 2
    dash, gap = (16, 7) if hot else (7, 11)
    offset = ar["t"] * (190 if hot else 60)

    line_surf = pygame.Surface((W, H), pygame.SRCALPHA)
    draw_dashed_line(line_surf, with_alpha(color, alpha * 255), (ar["ox"], ar["oy"]), (ex, ey), width, dash, gap, offset)
    screen.blit(line_surf, (0, 0))

    bow_alpha = 0.34 + p * 0.5
    pull = 4 + p * 13
    bow_surf = pygame.Surface((W, H), pygame.SRCALPHA)
    bow_col = with_alpha(color, bow_alpha * 255)

    arc_pts = []
    for i in range(17):
        ang = 2.35 + (3.93 - 2.35) * i / 16
        lx = pull + 15 * math.cos(ang)
        ly = 15 * math.sin(ang)
        arc_pts.append(rotate_translate(lx, ly, ar["a"], ar["ox"], ar["oy"]))
    if len(arc_pts) > 1:
        pygame.draw.lines(bow_surf, bow_col, False, arc_pts, 2)

    p1 = rotate_translate(pull - 10.6, -10.6, ar["a"], ar["ox"], ar["oy"])
    p2 = rotate_translate(pull - 20, 0, ar["a"], ar["ox"], ar["oy"])
    p3 = rotate_translate(pull - 10.6, 10.6, ar["a"], ar["ox"], ar["oy"])
    pygame.draw.lines(bow_surf, bow_col, False, [p1, p2, p3], 1)
    screen.blit(bow_surf, (0, 0))


def draw_pad():
    p = pad["hold"]
    puff = 1 + math.sin(pad["phase"] * 2.1) * 0.028 + p * 0.07
    r = pad["r"] * puff

    fill_alpha = 0.09 + p * 0.26
    ring_alpha = 0.5 + p * 0.3
    surf = pygame.Surface((W, H), pygame.SRCALPHA)
    pygame.draw.circle(surf, with_alpha(CARMINE, fill_alpha * 255), (int(pad["x"]), int(pad["y"])), int(r))
    ring_col = CARMINE if pad["inside"] else PLUM
    _draw_dashed_circle(surf, with_alpha(ring_col, ring_alpha * 255), (pad["x"], pad["y"]), r, 2, 6, 9, -pad["phase"] * 14)

    if p > 0.002:
        rect = pygame.Rect(pad["x"] - r, pad["y"] - r, r * 2, r * 2)
        start_ang = -math.pi / 2
        end_ang = start_ang + p * 2 * math.pi
        pygame.draw.arc(surf, CARMINE, rect, start_ang, end_ang, 4)

    screen.blit(surf, (0, 0))

    label = serif(max(14, int(pad["r"] * 0.44)))
    txt = label.render("start", True, INK)
    screen.blit(txt, (pad["x"] - txt.get_width() / 2, pad["y"] - txt.get_height() / 2 + 1))

    hint_font = sans(13)
    hint_str = "hold it there…" if pad["inside"] else "bring your heart here"
    hint_surf = hint_font.render(hint_str, True, INK_SOFT)
    hint_surf.set_alpha(int(0.75 * 255))
    screen.blit(hint_surf, (pad["x"] - hint_surf.get_width() / 2, pad["y"] + r + 12))


def _draw_dashed_circle(surface, color, center, radius, width, dash_len, gap_len, offset):
    if radius <= 0:
        return
    circumference = 2 * math.pi * radius
    period = dash_len + gap_len
    n_periods = max(1, int(circumference / period))
    pos = -offset % period
    ang0 = -pos / radius
    steps_per_dash = 6
    a = ang0
    total_ang = 2 * math.pi + abs(ang0)
    travelled = 0.0
    ang = ang0
    while travelled < circumference + period:
        dash_ang = dash_len / radius
        pts = []
        for i in range(steps_per_dash + 1):
            aa = ang + dash_ang * i / steps_per_dash
            pts.append((center[0] + radius * math.cos(aa), center[1] + radius * math.sin(aa)))
        pygame.draw.lines(surface, color, False, pts, width)
        ang += dash_ang + (gap_len / radius)
        travelled += period


def draw_hud():
    if mode == TITLE:
        return
    pop = 1 + score_pop * 0.22
    score_font = serif(max(20, int(54 * pop)), bold=False)
    score_surf = score_font.render(str(score), True, CARMINE)
    screen.blit(score_surf, (26, 74 - score_surf.get_height() + 12))

    label_font = sans(14)
    label_surf = label_font.render("saves", True, INK_SOFT)
    screen.blit(label_surf, (28, 94 - label_surf.get_height() + 8))

    round_label = label_font.render(f"round {level}", True, INK_SOFT)
    screen.blit(round_label, (W - 26 - round_label.get_width(), 94 - round_label.get_height() + 8))

    level_font = serif(40, bold=False)
    level_surf = level_font.render(str(level), True, INK)
    screen.blit(level_surf, (W - 26 - level_surf.get_width(), 74 - level_surf.get_height() + 12))


def draw_rings():
    for r in rings:
        alpha = max(0.0, r["life"]) * 0.55
        draw_circle_alpha(screen, with_alpha(CARMINE, alpha * 255), (int(r["x"]), int(r["y"])), int(r["r"]), max(1, int(r["w"])))


def draw_bits():
    for b in bits:
        alpha = max(0.0, b["life"])
        surf = pygame.Surface((60, 60), pygame.SRCALPHA)
        cx, cy = 30, 30
        pts_local = [(0, -b["size"]), (b["size"] * 0.8, b["size"] * 0.4), (-b["size"] * 0.6, b["size"] * 0.7)]
        pts = [rotate_translate(px, py, b["rot"], cx, cy) for px, py in pts_local]
        pygame.draw.polygon(surf, with_alpha(CARMINE, alpha * 255), pts)
        screen.blit(surf, (b["x"] - 30, b["y"] - 30))

retry_button_rect = pygame.Rect(0, 0, 0, 0)
sound_button_rect = pygame.Rect(0, 0, 0, 0)


def draw_button(rect, text, hovered):
    color = CARMINE if hovered else INK
    pygame.draw.rect(screen, color, rect, border_radius=999)
    f = sans(17, bold=True)
    t = f.render(text, True, PAPER)
    screen.blit(t, (rect.centerx - t.get_width() / 2, rect.centery - t.get_height() / 2))


def draw_title_screen(mouse_pos):
    pad_left = max(28, int(W * 0.07))
    x = pad_left
    y = H * 0.5 - 140

    h1_size = max(40, min(int(W * 0.09), 100))
    h1 = serif(h1_size, bold=True)
    line1 = h1.render("Save my", True, CARMINE)
    line2 = h1.render("heart", True, INK)
    screen.blit(line1, (x, y))
    y += line1.get_height() * 0.92
    screen.blit(line2, (x + line1.get_width() * 0.1, y))
    y += line2.get_height() * 1.15

    blurb_font = sans(17)
    blurb_text = "Drag the heart with your mouse. A dotted line shows where the next arrow will fly — get off that line before it fires. Every arrow you dodge is one more save."
    for line in wrap_text(blurb_text, blurb_font, min(420, W - x - 40)):
        s = blurb_font.render(line, True, INK_SOFT)
        screen.blit(s, (x, y))
        y += s.get_height() * 1.35
    y += 8

    hint_font = sans(17)
    hint_text = "Rest your heart on start to begin."
    for line in wrap_text(hint_text, hint_font, min(420, W - x - 40)):
        s = hint_font.render(line, True, PLUM)
        screen.blit(s, (x, y))
        y += s.get_height() * 1.35


def draw_over_screen(mouse_pos):
    global retry_button_rect
    pad_left = max(28, int(W * 0.07))
    x = pad_left
    y = H * 0.5 - 130

    h2_size = max(36, min(int(W * 0.08), 86))
    h2 = serif(h2_size, bold=True)
    s = h2.render("Heart broken", True, INK)
    screen.blit(s, (x, y))
    y += s.get_height() * 1.25

    num_font = serif(56, bold=False)
    quiet_font = serif(32, bold=False)
    cap_font = sans(15)

    score_num = num_font.render(str(score), True, CARMINE)
    screen.blit(score_num, (x, y))
    cap1 = cap_font.render("saves", True, INK_SOFT)
    screen.blit(cap1, (x, y + score_num.get_height() + 6))

    bx = x + score_num.get_width() + 40
    best_num = quiet_font.render(str(best), True, INK_SOFT)
    screen.blit(best_num, (bx, y + (score_num.get_height() - best_num.get_height())))
    cap2 = cap_font.render("best", True, INK_SOFT)
    screen.blit(cap2, (bx, y + score_num.get_height() + 6))

    y += max(score_num.get_height(), best_num.get_height()) + 40

    broken_font = sans(17)
    broken_surf = broken_font.render(f"Hearts broken: {broken}", True, INK_SOFT)
    screen.blit(broken_surf, (x, y))
    y += broken_surf.get_height() + 30

    btn_w, btn_h = 190, 54
    retry_button_rect = pygame.Rect(x, y, btn_w, btn_h)
    hovered = retry_button_rect.collidepoint(mouse_pos)
    draw_button(retry_button_rect, "Try again", hovered)


def draw_sound_button(mouse_pos):
    global sound_button_rect
    f = sans(11, bold=True)
    label = "SOUND OFF" if muted else "SOUND ON"
    text_surf = f.render(label, True, INK_SOFT if not muted else INK_SOFT)
    pad_x, pad_y = 18, 10
    w = text_surf.get_width() + pad_x * 2
    h = text_surf.get_height() + pad_y * 2
    rect = pygame.Rect(W - w - 24, H - h - 24, w, h)
    sound_button_rect = rect
    hovered = rect.collidepoint(mouse_pos)
    color = CARMINE if hovered else INK_SOFT
    alpha = 255 if hovered else int(0.65 * 255) if not muted else int(0.4 * 255)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(surf, with_alpha(color, alpha), surf.get_rect(), width=1, border_radius=999)
    screen.blit(surf, (rect.x, rect.y))
    ts = f.render(label, True, color)
    ts.set_alpha(alpha)
    screen.blit(ts, (rect.centerx - ts.get_width() / 2, rect.centery - ts.get_height() / 2))

last_time = pygame.time.get_ticks() / 1000.0
running = True

while running:
    now_ticks = pygame.time.get_ticks() / 1000.0
    dt = min(now_ticks - last_time, 0.05)
    last_time = now_ticks
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            resize(event.w, event.h)
        elif event.type == pygame.MOUSEMOTION:
            heart["tx"], heart["ty"] = event.pos
        elif event.type == pygame.MOUSEBUTTONDOWN:
            heart["tx"], heart["ty"] = event.pos
            if sound_button_rect.collidepoint(event.pos):
                muted = not muted
            elif mode == OVER and retry_button_rect.collidepoint(event.pos):
                start_game()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
                muted = not muted
            elif event.key in (pygame.K_SPACE, pygame.K_RETURN) and mode != PLAYING:
                start_game()
            elif event.key == pygame.K_ESCAPE:
                running = False

    k = 1 - math.pow(0.0008, dt)
    heart["x"] += (heart["tx"] - heart["x"]) * k
    heart["y"] += (heart["ty"] - heart["y"]) * k
    heart["beat"] += dt * (4.4 + level * 0.42)
    score_pop = max(0.0, score_pop - dt * 3.4)
    field_tint = max(0.0, field_tint - dt * 1.6)
    shake = max(0.0, shake - dt * 46)

    if mode == TITLE:
        pad["phase"] += dt
        near = math.hypot(heart["x"] - pad["x"], heart["y"] - pad["y"]) < pad["r"] - 4
        pad["inside"] = near
        if near:
            pad["hold"] = min(1.0, pad["hold"] + dt / PAD_TIME)
            step = int(pad["hold"] * 3)
            if step > pad["tick"]:
                pad["tick"] = step
                sfx_tick(pad["x"], step)
            if pad["hold"] >= 1.0:
                start_game()
        else:
            pad["hold"] = max(0.0, pad["hold"] - dt / 0.32)
            if pad["hold"] == 0:
                pad["tick"] = 0

    if mode == PLAYING:
        d = difficulty()
        spawn_timer -= dt
        if spawn_timer <= 0 and len(arrows) < 14:
            spawn_arrow(d)
            if random.random() < d["two"]:
                spawn_arrow(d)
            if random.random() < d["three"]:
                spawn_arrow(d)
            spawn_timer = d["every"]

    for ar in arrows[:]:
        ar["t"] += dt
        if ar["state"] == "warn":
            if ar["t"] >= ar["warn"]:
                ar["state"] = "fly"
                ar["t"] = 0.0
                sfx_arrow(ar["ox"], now_ticks)
        else:
            ar["x"] += ar["dx"] * ar["speed"] * dt
            ar["y"] += ar["dy"] * ar["speed"] * dt

            if mode == PLAYING and heart["alive"]:
                hit = False
                for s in range(3):
                    px = ar["x"] - ar["dx"] * (s * 13)
                    py = ar["y"] - ar["dy"] * (s * 13)
                    dist = math.hypot(px - heart["x"], py - heart["y"])
                    if dist < heart["r"] + 4:
                        end_game()
                        hit = True
                        break
                    if not ar["grazed"] and dist < heart["r"] + 30:
                        ar["grazed"] = True
                        sfx_graze(heart["x"], now_ticks)
                        rings.append({"x": heart["x"], "y": heart["y"], "r": heart["r"] + 6, "life": 1.0, "w": 2})
                if hit:
                    continue

            margin = 140
            if ar["x"] < -margin or ar["x"] > W + margin or ar["y"] < -margin or ar["y"] > H + margin:
                arrows.remove(ar)
                if mode == PLAYING:
                    score += 1
                    score_pop = 1.0
                    sfx_dodge(ar["x"], score)
                    rings.append({"x": heart["x"], "y": heart["y"], "r": heart["r"] + 4, "life": 1.0, "w": 2.6})
                continue

    for r in rings[:]:
        r["life"] -= dt * 2.2
        r["r"] += dt * 72
        if r["life"] <= 0:
            rings.remove(r)

    for b in bits[:]:
        b["vy"] += 900 * dt
        b["x"] += b["vx"] * dt
        b["y"] += b["vy"] * dt
        b["rot"] += b["vr"] * dt
        b["life"] -= dt * 0.75
        if b["life"] <= 0:
            bits.remove(b)

    draw_field()

    render_target = screen
    if shake > 0.2:
        offset = (rand(-shake, shake) * 0.5, rand(-shake, shake) * 0.5)
        temp = pygame.Surface((W, H))
        temp.blit(screen, (0, 0))
        screen.fill(PAPER)
        screen.blit(temp, offset)

    draw_rings()

    for ar in arrows:
        if ar["state"] == "warn":
            draw_warning(ar)
        else:
            draw_arrow(ar)

    draw_bits()

    if mode == TITLE:
        draw_pad()

    draw_heart_shape()
    draw_hud()

    if mode == TITLE:
        draw_title_screen(mouse_pos)
    elif mode == OVER:
        draw_over_screen(mouse_pos)

    draw_sound_button(mouse_pos)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
