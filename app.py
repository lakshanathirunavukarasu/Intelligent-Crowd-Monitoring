"""
Crowd Counting App (Streamlit) - redesigned UI
Models: fast_csrnet.pth (CSRNet) and fast_mcnn.pth (MCNN)

Run:
    pip install -r requirements.txt
    streamlit run app.py

Put app.py, fast_csrnet.pth and fast_mcnn.pth in the same folder.
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------
# Model definitions (rebuilt from the layer names / shapes in your .pth files)
# ----------------------------------------------------------------------------


class CountHead(nn.Sequential):
    """count_head.0 = pool, .1 = flatten, .2 = Linear, .3 = ReLU, .4 = Linear"""

    def __init__(self, ch=32):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch),
            nn.ReLU(inplace=True),
            nn.Linear(ch, 1),
        )


class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        # VGG-style frontend (stride 4): indices 0,2,5,7,10 are convs
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),      # 0,1
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),     # 2,3
            nn.MaxPool2d(2, 2),                                         # 4
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),    # 5,6
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),   # 7,8
            nn.MaxPool2d(2, 2),                                         # 9
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),   # 10,11
        )
        # Dilated backend: indices 0,2,4,6,8 are convs (dilation=2 as in CSRNet paper)
        d = dict(padding=2, dilation=2)
        self.backend = nn.Sequential(
            nn.Conv2d(256, 256, 3, **d), nn.ReLU(inplace=True),         # 0,1
            nn.Conv2d(256, 256, 3, **d), nn.ReLU(inplace=True),         # 2,3
            nn.Conv2d(256, 128, 3, **d), nn.ReLU(inplace=True),         # 4,5
            nn.Conv2d(128, 64, 3, **d), nn.ReLU(inplace=True),          # 6,7
            nn.Conv2d(64, 32, 3, **d), nn.ReLU(inplace=True),           # 8,9
        )
        self.density_head = nn.Conv2d(32, 1, 1)
        self.count_head = CountHead(32)

    def forward(self, x):
        feat = self.backend(self.frontend(x))
        density = self.density_head(feat)  # raw; ReLU applied in infer()
        count = self.count_head(feat)
        return density, count


def _col(k):
    p = k // 2
    return nn.Sequential(
        nn.Conv2d(3, 16, k, padding=p), nn.ReLU(inplace=True),    # 0,1
        nn.MaxPool2d(2),                                          # 2
        nn.Conv2d(16, 32, k, padding=p), nn.ReLU(inplace=True),   # 3,4
    )


class MCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.column1 = _col(5)
        self.column2 = _col(7)
        self.column3 = _col(9)
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1), nn.ReLU(inplace=True),   # 0,1
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),   # 2,3
        )
        self.density_head = nn.Conv2d(32, 1, 1)
        self.count_head = CountHead(32)

    def forward(self, x):
        feat = torch.cat(
            [self.column1(x), self.column2(x), self.column3(x)], dim=1
        )
        feat = self.fusion(feat)
        density = self.density_head(feat)  # raw; ReLU applied in infer()
        count = self.count_head(feat)
        return density, count


class HybridMCNNCSRNet(nn.Module):
    """Hybrid = MCNN branch + CSRNet branch + learned `fusion` conv stack.

    Your checkpoint has mcnn.*, csrnet.* and fusion.0/2/4/6 layers. The fusion layers are built
    automatically from the shapes stored in the .pth file (see from_state), and the fusion input is
    chosen from the number of input channels of fusion.0:
        2   -> the two density maps (MCNN, CSRNet)
        64  -> the two 32-channel feature maps
        128 -> MCNN 96-channel column features + CSRNet 32-channel features
    """

    def __init__(self):
        super().__init__()
        self.mcnn = MCNN()
        self.csrnet = CSRNet()
        self.fusion = nn.Identity()
        self.out_mode = "auto"   # auto | fusion | branch_avg | residual | feature_map
        self.last_used = "-"
        self.fuse_to = "csrnet"  # resolution of the fused map: "csrnet" (stride 4) or "mcnn" (stride 2)

    @classmethod
    def from_state(cls, state):
        m = cls()
        idx = sorted({int(k.split(".")[1]) for k in state
                      if k.startswith("fusion.") and k.endswith(".weight")})
        if idx:
            mods = []
            for i in range(max(idx) + 1):
                if i in idx:
                    o, c_in, kh, kw = state[f"fusion.{i}.weight"].shape
                    mods.append(nn.Conv2d(c_in, o, (kh, kw), padding=(kh // 2, kw // 2)))
                else:
                    mods.append(nn.ReLU(inplace=True))
            m.fusion = nn.Sequential(*mods)
        return m

    def _parts(self, x):
        mc, cs = self.mcnn, self.csrnet
        cols = torch.cat([mc.column1(x), mc.column2(x), mc.column3(x)], dim=1)   # 96 ch, stride 2
        f1 = mc.fusion(cols)                                                      # 32 ch, stride 2
        f2 = cs.backend(cs.frontend(x))                                           # 32 ch, stride 4
        d1, d2 = mc.density_head(f1), cs.density_head(f2)
        c1, c2 = mc.count_head(f1), cs.count_head(f2)
        in_ch = self.fusion[0].in_channels
        options = {2: (d1, d2), 64: (f1, f2), 128: (cols, f2)}
        if in_ch not in options:
            raise ValueError(f"fusion.0 expects {in_ch} input channels - not one of the supported "
                             f"layouts {sorted(options)}. Send the layer list so the class can be adjusted.")
        u, v = options[in_ch]
        size = d2.shape[-2:] if self.fuse_to == "csrnet" else d1.shape[-2:]
        u = F.interpolate(u, size=size, mode="bilinear", align_corners=False)
        v = F.interpolate(v, size=size, mode="bilinear", align_corners=False)
        fused = self.fusion(torch.cat([u, v], dim=1))
        return d1, d2, c1, c2, fused, f2

    @staticmethod
    def _branch_avg(d1, d2):
        """Average of the two branch density maps at CSRNet size (counts preserved)."""
        ratio = (d1.shape[-2] * d1.shape[-1]) / (d2.shape[-2] * d2.shape[-1])
        d1r = F.interpolate(d1, size=d2.shape[-2:], mode="bilinear", align_corners=False) * ratio
        return (d1r + d2) / 2

    def forward(self, x):
        d1, d2, c1, c2, fused, f2 = self._parts(x)
        count = ((c1 + c2) / 2).clamp(min=0.0)
        avg = self._branch_avg(d1, d2)
        fused1 = fused[:, :1]

        def feature_density():
            # last resort: CSRNet feature activity, rescaled so its sum equals the predicted count
            fm = F.relu(f2).mean(dim=1, keepdim=True)
            fm = fm / (fm.sum(dim=(1, 2, 3), keepdim=True) + 1e-8)
            return fm * count.view(-1, 1, 1, 1)

        need = 0.05 * count.clamp(min=1.0).sum()
        mode = self.out_mode
        if mode == "fusion":
            density, used = fused1, "fusion"
        elif mode == "branch_avg":
            density, used = avg, "branch_avg"
        elif mode == "residual":
            density = avg + F.interpolate(fused1, size=avg.shape[-2:], mode="bilinear", align_corners=False)
            used = "residual"
        elif mode == "feature_map":
            density, used = feature_density(), "feature_map (approx.)"
        else:  # auto: fusion -> branch average -> feature map, first one that is not empty after ReLU
            if bool(F.relu(fused1).sum() > need):
                density, used = fused1, "fusion"
            elif bool(F.relu(avg).sum() > need):
                density, used = avg, "branch_avg"
            else:
                density, used = feature_density(), "feature_map (approx.)"
        self.last_used = used
        return density, count

    @torch.no_grad()
    def debug(self, x):
        d1, d2, c1, c2, fused, _ = self._parts(x)
        return {"MCNN density (raw)": d1.sum().item(), "MCNN density (ReLU)": F.relu(d1).sum().item(),
                "CSRNet density (raw)": d2.sum().item(), "CSRNet density (ReLU)": F.relu(d2).sum().item(),
                "MCNN count head": c1.sum().item(), "CSRNet count head": c2.sum().item(),
                "fusion (raw)": fused.sum().item(), "fusion (ReLU)": F.relu(fused).sum().item(),
                "fusion channels": int(fused.shape[1])}


CODE_STAMP = os.path.getmtime(os.path.abspath(__file__))

MODELS = {
    "CSRNet (fast_csrnet.pth)": (CSRNet, "fast_csrnet.pth"),
    "MCNN (fast_mcnn.pth)": (MCNN, "fast_mcnn.pth"),
    "Hybrid (hybrid_mcnn_csrnet_best.pth)": (HybridMCNNCSRNet, "hybrid_mcnn_csrnet_best.pth"),
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

MAX_SIDE = 1600  # long side of the working canvas (keeps heatmaps sharp but fast)
COUNT_SOURCES = ["Density map sum", "Count head output", "Average of both"]


def resolve_weights(filename: str) -> str:
    """Find a weights file next to app.py. Tolerates .pt/.pth, double extensions and case."""
    folder = os.path.dirname(os.path.abspath(__file__))
    exact = os.path.join(folder, filename)
    if os.path.exists(exact):
        return exact
    stem = os.path.splitext(filename)[0].lower()
    found = [f for f in os.listdir(folder) if f.lower().endswith((".pth", ".pt", ".pth.pth", ".pt.pth"))]
    close = [f for f in found if f.lower().startswith(stem)]
    if len(close) == 1:
        return os.path.join(folder, close[0])
    listing = ", ".join(sorted(found)) if found else "no .pth / .pt files"
    raise FileNotFoundError(
        f"Weights not found: {exact}. Files in the app folder ({folder}): {listing}. "
        "Rename the file to match exactly, or put it next to app.py.")


@st.cache_resource(show_spinner="Loading model...")
def load_model(name: str, code_stamp: float = 0.0):
    """code_stamp = app.py modification time, so edited model code never uses an old cached model."""
    cls, path = MODELS[name]
    path = resolve_weights(path)
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    image_size = int(ckpt.get("image_size", 256)) if isinstance(ckpt, dict) else 256
    model = cls.from_state(state) if hasattr(cls, "from_state") else cls()
    model.load_state_dict(state, strict=True)  # fails loudly if architecture differs
    model.to(DEVICE).eval()
    return model, image_size


def inspect_checkpoint(filename: str):
    """Return (top-level info dict, DataFrame of layer names / shapes) for a .pth file."""
    path = resolve_weights(filename)
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    info = {k: (v if isinstance(v, (int, float, str)) else type(v).__name__)
            for k, v in ckpt.items()} if isinstance(ckpt, dict) else {}
    rows = [{"layer name": k, "shape": str(tuple(v.shape)), "dtype": str(v.dtype).replace("torch.", "")}
            for k, v in state.items() if hasattr(v, "shape")]
    return info, pd.DataFrame(rows)


NORM_MODES = ["ImageNet (/255, mean/std)", "0-1 (/255 only)", "-1 to 1", "Raw 0-255"]


def tile_tensor(pil: Image.Image, size: int, mode: str) -> torch.Tensor:
    arr = np.asarray(pil.resize((size, size), Image.BICUBIC), dtype=np.float32)
    if mode == NORM_MODES[0]:
        arr = (arr / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    elif mode == NORM_MODES[1]:
        arr = arr / 255.0
    elif mode == NORM_MODES[2]:
        arr = arr / 127.5 - 1.0
    # NORM_MODES[3]: raw 0-255, unchanged
    return torch.from_numpy(arr).permute(2, 0, 1)


@torch.no_grad()
def infer(model, size, pil_img, grid, use_norm, flip_tta, relu_density=True):
    """Returns (working image, full-res density map, density sum, count-head sum).

    grid=1  -> whole image resized to `size` (same as training).
    grid>1  -> image split into grid x grid tiles, each tile resized to `size`;
               counts of all tiles are added. Helps on large / very dense images.
    The density map is upsampled with bicubic and rescaled so its sum is
    EXACTLY the model's predicted count (no count lost by resizing).
    """
    img = pil_img.convert("RGB")
    w0, h0 = img.size
    s = min(1.0, MAX_SIDE / max(w0, h0))
    if s < 1.0:
        img = img.resize((round(w0 * s), round(h0 * s)), Image.LANCZOS)
    W, H = img.size
    grid = max(1, min(grid, W // 64, H // 64))  # avoid absurdly tiny tiles

    xs = np.linspace(0, W, grid + 1).round().astype(int)
    ys = np.linspace(0, H, grid + 1).round().astype(int)
    boxes = [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(grid) for i in range(grid)]

    batch = torch.stack([tile_tensor(img.crop(b), size, use_norm) for b in boxes]).to(DEVICE)
    d, c = model(batch)
    if flip_tta:  # average with horizontally-flipped prediction
        d2, c2 = model(batch.flip(-1))
        d, c = (d + d2.flip(-1)) / 2, (c + c2) / 2
    if relu_density:
        d = F.relu(d)

    canvas = torch.zeros(H, W)
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        dk = d[k:k + 1].float().cpu()
        total = dk.sum()
        up = F.interpolate(dk, size=(y1 - y0, x1 - x0), mode="bicubic", align_corners=False)
        if relu_density:
            up = up.clamp(min=0)
        up = up * (total / (up.sum() + 1e-8))  # keep the tile's count unchanged
        canvas[y0:y1, x0:x1] = up[0, 0]
    return img, canvas.numpy(), float(canvas.sum()), float(c.sum().item())


def pick_count(c_sum, c_head, source):
    if source == "Density map sum":
        return c_sum
    if source == "Count head output":
        return c_head
    return (c_sum + c_head) / 2


def gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    r = max(1, min(int(3 * sigma), min(arr.shape) - 1))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return t[0, 0].numpy()


def render_heatmap(img: Image.Image, dens: np.ndarray, alpha: float, smooth: float):
    """Full-resolution heatmap + overlay. Empty areas stay transparent."""
    dens = gaussian_blur(dens, smooth)
    vmax = max(np.percentile(dens, 99.5), 0.2 * dens.max(), 1e-8)  # robust scale
    d = np.clip(dens / vmax, 0, 1)
    rgb = (colormaps["turbo"](d)[..., :3] * 255).astype(np.float32)
    a = (np.clip(d * 2.0, 0, 1) * alpha)[..., None]  # more crowd -> more colour
    base = np.asarray(img, dtype=np.float32)
    overlay = base * (1 - a) + rgb * a
    return Image.fromarray(rgb.astype(np.uint8)), Image.fromarray(overlay.astype(np.uint8))


def short_name(m):
    return m.split(" ")[0]


# ----------------------------------------------------------------------------
# XAI + crowd-safety helpers
# ----------------------------------------------------------------------------

LEVELS = [
    ("Normal", "🟢", "#22C55E"),
    ("Moderate", "🟡", "#EAB308"),
    ("High", "🟠", "#F97316"),
    ("Severe", "🔴", "#EF4444"),
    ("Critical", "🚨", "#B91C1C"),
]

RECS = {
    0: [("✅", "The crowd is small and comfortable. No action is needed."),
        ("🚶", "People can move freely, so you can continue as normal."),
        ("👀", "Check again after some time to see if more people are arriving.")],
    1: [("👀", "The crowd is getting bigger. Keep an eye on the entrance and exit areas."),
        ("🚪", "Keep all walkways and exits clear so people can move easily."),
        ("📢", "Politely ask people to keep moving and not to block the path.")],
    2: [("🚶", "Avoid this area for now, or visit a little later when it is less crowded."),
        ("🧍", "Slow down the number of people entering by letting them in a few at a time."),
        ("🚪", "Use all the available gates and exits, not just one."),
        ("📢", "Ask people to spread out and move to less crowded areas."),
        ("🧒", "Keep children and elderly people close, and stay with your group.")],
    3: [("🛑", "Please stop more people from entering this area for now."),
        ("↔️", "Guide people towards the quieter zones and open exits."),
        ("🧭", "Keep a clear one-way path so people do not push against each other."),
        ("👮", "Keep staff or volunteers near the busiest zones to help people."),
        ("🧒", "Keep children and elderly people away from the crowded zones.")],
    4: [("🛑", "The area is too crowded. Stop letting new people in until it clears."),
        ("🚪", "Open all the exits and guide people out slowly and calmly."),
        ("🗣️", "Use clear and calm announcements. Ask people not to push or run."),
        ("👮", "Keep staff at the busiest zones and ask for help from the organisers."),
        ("🔁", "Check the count again after a few minutes to confirm the crowd is reducing.")],
}


def risk_level(density, T):
    """Crowd level 0-4 (Normal..Critical). `density` and T are in the same unit (people)."""
    r = density / max(T, 1e-6)
    if r < 0.5:
        return 0
    if r < 1.0:
        return 1
    if r < 2.0:
        return 2
    if r < 2.75:
        return 3
    return 4


def coarse_sum(dens, n=12):
    """Sum a density map into an n x n grid (total count preserved exactly)."""
    H, W = dens.shape
    ys = np.linspace(0, H, n + 1).round().astype(int)[:-1]
    xs = np.linspace(0, W, n + 1).round().astype(int)[:-1]
    return np.add.reduceat(np.add.reduceat(dens, ys, axis=0), xs, axis=1).astype(np.float64)


def to_zones(cells, n):
    k = cells.shape[0] // n
    return cells.reshape(n, k, n, k).sum(axis=(1, 3))


def zone_name(r, c, n):
    if n == 3:
        if r == 1 and c == 1:
            return "centre"
        return f"{['top', 'middle', 'bottom'][r]}-{['left', 'centre', 'right'][c]}"
    if n == 2:
        return f"{['top', 'bottom'][r]}-{['left', 'right'][c]}"
    return f"R{r + 1}-C{c + 1}"


def _font(size):
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans:bold"), size)
    except Exception:
        return ImageFont.load_default()


def zone_overlay(img, zc, area_m2, T):
    """Draw an n x n grid on the image; each zone coloured by its crowd level."""
    n = zc.shape[0]
    W, H = img.size
    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    xs = np.linspace(0, W, n + 1).round().astype(int)
    ys = np.linspace(0, H, n + 1).round().astype(int)
    font = _font(max(14, min(W, H) // (n * 8)))
    zone_area = area_m2 / (n * n)
    for r in range(n):
        for c in range(n):
            d = zc[r, c] / zone_area
            hexcol = LEVELS[risk_level(d, T)][2].lstrip("#")
            col = tuple(int(hexcol[i:i + 2], 16) for i in (0, 2, 4))
            box = (int(xs[c]), int(ys[r]), int(xs[c + 1]), int(ys[r + 1]))
            dr.rectangle(box, fill=col + (75,), outline=col + (255,), width=3)
            dr.text((box[0] + 8, box[1] + 6), f"{zc[r, c]:.0f} people\n{LEVELS[risk_level(d, T)][0]}",
                    fill=(255, 255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return Image.alpha_composite(base, layer).convert("RGB")


def _work_image(pil_img):
    img = pil_img.convert("RGB")
    w0, h0 = img.size
    s = min(1.0, MAX_SIDE / max(w0, h0))
    if s < 1.0:
        img = img.resize((round(w0 * s), round(h0 * s)), Image.LANCZOS)
    return img


def gradcam(model, size, pil_img, norm_mode):
    """Grad-CAM for crowd counting: which image regions pushed the predicted count up.

    Target = sum of the (ReLU) density map. Features = last frontend layer (CSRNet) or the
    concatenated multi-column features (MCNN), i.e. before the layers that produce density.
    Returns (working image, cam in 0-1 at working-image resolution).
    """
    img = _work_image(pil_img)
    W, H = img.size
    x = tile_tensor(img, size, norm_mode).unsqueeze(0).to(DEVICE)
    store = {}

    def grab(t):
        t.retain_grad()
        store["a"] = t

    if hasattr(model, "frontend"):
        h = model.frontend.register_forward_hook(lambda m, i, o: grab(o))
    elif hasattr(model, "fusion"):
        h = model.fusion.register_forward_pre_hook(lambda m, i: grab(i[0]))
    elif hasattr(model, "csrnet"):  # hybrid: explain through its CSRNet branch
        h = model.csrnet.frontend.register_forward_hook(lambda m, i, o: grab(o))
    else:
        raise ValueError("Grad-CAM is not available for this architecture.")
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            d, _ = model(x)
            F.relu(d).sum().backward()
    finally:
        h.remove()
    a = store["a"].detach()
    g = store["a"].grad
    if g is None:
        return img, np.zeros((H, W), dtype=np.float32)
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return img, cam.astype(np.float32)


def explain_text(sn, total, share):
    """Plain-English explanation built from the density distribution."""
    flat = np.sort(share.ravel())[::-1]
    cells50 = int(np.searchsorted(np.cumsum(flat), 0.5) + 1)
    area50 = cells50 / flat.size * 100
    z3 = to_zones(share, 3)
    r, c = np.unravel_index(int(z3.argmax()), z3.shape)
    pts = [f"The <b>{sn}</b> model estimates about <b>{total:.0f} people</b> in this image."]
    if area50 < 25:
        pts.append(f"Half of all people are packed into only <b>{area50:.0f}%</b> of the image area, "
                   "so the crowd is <b>highly concentrated</b>.")
    elif area50 < 45:
        pts.append(f"Half of all people are inside <b>{area50:.0f}%</b> of the image area, "
                   "so the crowd is <b>moderately concentrated</b>.")
    else:
        pts.append(f"Half of all people are spread over <b>{area50:.0f}%</b> of the image area, "
                   "so the crowd is <b>fairly evenly spread</b>.")
    pts.append(f"The busiest part is the <b>{zone_name(r, c, 3)}</b> area, "
               f"holding about <b>{z3[r, c] * 100:.0f}%</b> of the crowd.")
    pts.append("The Grad-CAM map shows where the model <i>looked</i>: bright red / yellow areas are the "
               "visual patterns (clusters of heads and shoulders) that pushed the count up the most.")
    return pts


def reliability(c_sum, c_head, other_gap=None):
    gap = abs(c_sum - c_head) / max((c_sum + c_head) / 2, 1.0)
    worst = max(gap, other_gap or 0.0)
    label = "High" if worst < 0.10 else ("Medium" if worst < 0.25 else "Low")
    return label, gap


# ----------------------------------------------------------------------------
# UI styling
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Crowd Counting", page_icon="👥", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

:root {
    --accent: #6C63FF;
    --accent2: #00C9A7;
    --accent3: #FF6B9D;
    --ink: #1E293B;
    --muted: #64748B;
    --card: #FFFFFF;
    --line: #E5E9F2;
}

/* ================= Global ================= */
html, body, .stApp, .stMarkdown, p, li, label, h1, h2, h3, h4, input, button, textarea {
    font-family: 'Poppins', sans-serif;
}
.stApp {
    background:
        radial-gradient(1200px 500px at 85% -10%, rgba(108,99,255,0.10), transparent 60%),
        radial-gradient(900px 400px at -10% 30%, rgba(0,201,167,0.08), transparent 60%),
        #F3F5FA;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1400px; }
footer { visibility: hidden; }

.stApp p, .stApp li, .stMarkdown, [data-testid="stCaptionContainer"] {
    font-size: 1.1rem !important; line-height: 1.6;
}
[data-testid="stCaptionContainer"] { font-size: 1.05rem !important; color: var(--muted); }
[data-testid="stWidgetLabel"] p { font-size: 1.1rem !important; font-weight: 500 !important; }

/* ================= Hero ================= */
.hero {
    background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 48%, #06B6A4 100%);
    border-radius: 26px; padding: 2.5rem 3rem; color: #fff;
    box-shadow: 0 18px 45px rgba(79, 70, 229, 0.35);
    margin-bottom: 1.8rem; position: relative; overflow: hidden;
}
.hero::before {
    content: ""; position: absolute; width: 380px; height: 380px; border-radius: 50%;
    right: -90px; top: -140px; background: rgba(255,255,255,0.10);
}
.hero::after {
    content: ""; position: absolute; width: 260px; height: 260px; border-radius: 50%;
    right: 180px; bottom: -150px; background: rgba(255,255,255,0.08);
}
.hero .eyebrow {
    display: inline-block; font-size: 0.95rem; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; background: rgba(255,255,255,0.18);
    padding: 0.3rem 0.9rem; border-radius: 999px; margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.7rem !important; font-weight: 800 !important; line-height: 1.2 !important;
    margin: 0 0 0.7rem 0 !important; padding: 0 !important; color: #fff !important;
    letter-spacing: -0.5px; max-width: 950px; position: relative; z-index: 1;
}
.hero p { font-size: 1.25rem !important; margin: 0; color: rgba(255,255,255,0.92) !important;
          position: relative; z-index: 1; }
.hero .chips { margin-top: 1.3rem; position: relative; z-index: 1; }
.chip {
    display: inline-block; background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.38); padding: 0.4rem 1.1rem; border-radius: 999px;
    font-size: 1rem; font-weight: 500; margin: 0 0.6rem 0.4rem 0; backdrop-filter: blur(6px);
}

/* ================= Section titles ================= */
.section-title {
    display: flex; align-items: center; gap: 0.7rem;
    font-size: 1.8rem; font-weight: 700; color: var(--ink); margin: 2.2rem 0 0.9rem 0;
}
.section-title::before {
    content: ""; width: 7px; height: 1.9rem; border-radius: 6px;
    background: linear-gradient(180deg, var(--accent), var(--accent2));
}
.section-sub { color: var(--muted); font-size: 1.05rem !important; margin: -0.5rem 0 1rem 0; }

/* ================= Bordered containers = cards ================= */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card); border: 1px solid var(--line) !important;
    border-radius: 22px !important; padding: 0.6rem 0.8rem;
    box-shadow: 0 8px 28px rgba(30, 41, 59, 0.07);
}

/* ================= Stat cards ================= */
.stat-card {
    background: var(--card); border: 1px solid var(--line); border-radius: 22px;
    padding: 1.5rem 1.7rem; position: relative; overflow: hidden;
    box-shadow: 0 8px 28px rgba(30, 41, 59, 0.07);
    transition: transform .2s ease, box-shadow .2s ease; height: 100%;
}
.stat-card::before {
    content: ""; position: absolute; left: 0; top: 0; right: 0; height: 6px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.stat-card:hover { transform: translateY(-5px); box-shadow: 0 16px 36px rgba(108,99,255,0.22); }
.stat-icon {
    width: 56px; height: 56px; border-radius: 16px; display: flex; align-items: center;
    justify-content: center; font-size: 1.7rem; margin-bottom: 0.8rem;
    background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(0,201,167,0.15));
}
.stat-label { font-size: 1.05rem; color: var(--muted); font-weight: 500; }
.stat-value {
    font-size: 2.9rem; font-weight: 800; line-height: 1.15; color: var(--ink);
}

/* ================= Empty state ================= */
.empty-card {
    background: var(--card); border: 2px dashed #C7CCF5; border-radius: 24px;
    padding: 2rem; text-align: center; margin-top: 1rem;
}
.empty-card .big { font-size: 3.2rem; }
.empty-card h3 { font-size: 1.7rem; margin: 0.3rem 0; color: var(--ink); }
.empty-card p { font-size: 1.15rem; color: var(--muted); margin: 0; }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; margin-top: 1.4rem; }
.step {
    background: var(--card); border: 1px solid var(--line); border-radius: 20px;
    padding: 1.4rem; box-shadow: 0 8px 24px rgba(30,41,59,0.06);
}
.step .num {
    width: 42px; height: 42px; border-radius: 50%; color: #fff; font-weight: 700;
    display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2)); margin-bottom: 0.7rem;
}
.step h4 { margin: 0 0 0.3rem 0; font-size: 1.25rem; color: var(--ink); }
.step p { margin: 0; color: var(--muted); font-size: 1.05rem !important; }

/* ================= Buttons ================= */
.stButton > button, .stDownloadButton > button {
    font-size: 1.15rem !important; font-weight: 600 !important;
    padding: 0.75rem 1.6rem !important; border-radius: 14px !important;
    border: 1px solid var(--line) !important; background: #fff; color: var(--ink);
    box-shadow: 0 4px 14px rgba(30,41,59,0.06); transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px); border-color: var(--accent) !important; color: var(--accent);
    box-shadow: 0 10px 24px rgba(108,99,255,0.28);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #4F46E5, #7C3AED) !important; color: #fff !important;
    border: none !important; padding: 0.95rem 1.6rem !important; font-size: 1.25rem !important;
    box-shadow: 0 12px 28px rgba(79,70,229,0.42);
}
.stButton > button[kind="primary"]:hover { color: #fff !important; filter: brightness(1.08); }

/* ================= File uploader ================= */
[data-testid="stFileUploader"] section {
    border: 2px dashed var(--accent) !important; border-radius: 18px !important;
    background: linear-gradient(135deg, rgba(108,99,255,0.06), rgba(0,201,167,0.06)) !important;
    padding: 1.6rem !important;
}
[data-testid="stFileUploader"] section small, [data-testid="stFileUploader"] section span {
    font-size: 1.1rem !important;
}
[data-testid="stFileUploaderFile"] { font-size: 1.05rem; }

/* ================= Expanders ================= */
[data-testid="stExpander"] {
    border: 1px solid var(--line) !important; border-radius: 18px !important;
    background: var(--card); box-shadow: 0 6px 20px rgba(30,41,59,0.06); margin-bottom: 0.9rem;
}
[data-testid="stExpander"] summary p { font-size: 1.2rem !important; font-weight: 600 !important; }

/* ================= Tables / images / progress ================= */
[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; border: 1px solid var(--line); }
[data-testid="stImage"] img { border-radius: 14px; box-shadow: 0 8px 22px rgba(0,0,0,0.18); }
[data-testid="stImageCaption"], figcaption { font-size: 1rem !important; color: var(--muted); }
.stProgress > div > div > div > div { background: linear-gradient(90deg, #6C63FF, #00C9A7) !important; }

.count-badge {
    display: inline-block; font-size: 1.5rem; font-weight: 800; color: #fff;
    background: linear-gradient(90deg, #FF6B9D, #6C63FF);
    padding: 0.4rem 1.3rem; border-radius: 14px; margin-bottom: 0.9rem;
    box-shadow: 0 8px 20px rgba(108,99,255,0.3);
}

/* ================= SIDEBAR (dark, professional) ================= */
[data-testid="stSidebar"] {
    background: linear-gradient(185deg, #0B1020 0%, #151A3C 55%, #251C5E 100%);
    border-right: none; box-shadow: 6px 0 30px rgba(15, 23, 42, 0.25);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
[data-testid="stSidebar"] svg { color: #CBD5E1; }

.brand {
    display: flex; align-items: center; gap: 0.9rem;
    padding: 1rem 0.4rem 1.2rem 0.4rem; margin-bottom: 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.12);
}
.brand .logo {
    width: 52px; height: 52px; border-radius: 16px; font-size: 1.7rem;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #6C63FF, #00C9A7);
    box-shadow: 0 8px 20px rgba(108,99,255,0.5);
}
.brand .name { font-size: 1.35rem; font-weight: 700; color: #fff; line-height: 1.2; }
.brand .tag { font-size: 0.9rem; color: #94A3B8; }

[data-testid="stSidebar"] h2 {
    color: #fff !important; font-size: 0.95rem !important; font-weight: 700 !important;
    letter-spacing: 1.8px; text-transform: uppercase;
    margin: 1.6rem 0 0.6rem 0 !important; padding: 0.55rem 0.9rem !important;
    border-radius: 12px; background: rgba(255,255,255,0.07);
    border-left: 4px solid var(--accent2);
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] label span,
[data-testid="stSidebar"] .stMarkdown p {
    color: #E2E8F0 !important; font-size: 1.08rem !important;
}

/* select + number inputs */
[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.18) !important; border-radius: 12px !important;
    color: #fff !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] * { color: #fff !important; font-size: 1.05rem !important; }
[data-testid="stSidebar"] input { color: #fff !important; background: transparent !important; font-size: 1.05rem !important; }
[data-testid="stSidebar"] [data-testid="stNumberInput"] button { background: rgba(255,255,255,0.10) !important; color: #fff !important; }
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg { color: #fff !important; }

/* slider */
[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"] { color: #5EEAD4 !important; font-weight: 600; }

/* radio / checkbox cards */
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px; padding: 0.45rem 0.8rem; margin-bottom: 0.4rem; width: 100%;
    transition: all .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(108,99,255,0.35); }
[data-testid="stSidebar"] [data-testid="stCheckbox"] {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px; padding: 0.5rem 0.8rem; margin-bottom: 0.4rem;
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.14) !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p { color: #fff !important; font-size: 1.1rem !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #94A3B8 !important; }

.device-badge {
    display: inline-block; padding: 0.5rem 1.2rem; border-radius: 999px;
    background: linear-gradient(90deg, #00C9A7, #6C63FF); color: #fff;
    font-weight: 600; font-size: 1.02rem; margin-top: 1.4rem;
    box-shadow: 0 8px 20px rgba(0,201,167,0.3);
}
.side-foot { margin-top: 1.2rem; color: #64748B; font-size: 0.9rem; }

/* ================= Crowd safety / XAI ================= */
.risk-banner {
    display: flex; align-items: center; gap: 1.4rem; padding: 1.5rem 2rem; border-radius: 24px;
    border: 2px solid var(--rc); background: color-mix(in srgb, var(--rc) 12%, white);
    box-shadow: 0 12px 30px rgba(30,41,59,0.10); margin: 0.6rem 0 1.2rem 0;
}
.rb-icon { font-size: 3.4rem; }
.rb-title { font-size: 1.9rem; font-weight: 800; color: var(--rc); line-height: 1.2; }
.rb-sub { font-size: 1.12rem; color: var(--muted); margin-top: 0.2rem; }
.rec-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1rem; margin: 0.8rem 0 0.4rem 0; }
.rec {
    display: flex; gap: 0.9rem; align-items: flex-start; background: #fff;
    border: 1px solid var(--line); border-left: 6px solid var(--rc); border-radius: 16px;
    padding: 1rem 1.2rem; box-shadow: 0 6px 18px rgba(30,41,59,0.06);
}
.rec .ri { font-size: 1.7rem; line-height: 1.2; }
.rec .rt { font-size: 1.08rem; color: var(--ink); line-height: 1.5; }
.legend { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.3rem 0 1rem 0; }
.lg { padding: 0.35rem 0.95rem; border-radius: 999px; color: #fff; font-weight: 600; font-size: 0.98rem; }
.xai-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.07), rgba(0,201,167,0.07));
    border: 1px solid var(--line); border-radius: 20px; padding: 1.2rem 1.6rem; margin: 1rem 0;
}
.xai-card h4 { margin: 0 0 0.5rem 0; font-size: 1.3rem; color: var(--ink); }
.xai-card ul { margin: 0; padding-left: 1.3rem; }
.xai-card li { font-size: 1.1rem; color: var(--ink); margin-bottom: 0.4rem; line-height: 1.55; }
.disclaimer { font-size: 0.95rem; color: var(--muted); border-top: 1px dashed var(--line); margin-top: 1rem; padding-top: 0.7rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def stat_card(icon, label, value):
    st.markdown(
        f'<div class="stat-card"><div class="stat-icon">{icon}</div>'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <span class="eyebrow">AI · Computer Vision</span>
        <h1>Deep learning based crowd density estimation and counting</h1>
        <div class="chips">
            <span class="chip">🧠 CSRNet</span>
            <span class="chip">⚡ MCNN</span>
            <span class="chip">🧬 Hybrid</span>
            <span class="chip">🔥 Density Heatmaps</span>
            <span class="chip">🔍 Explainable AI</span>
            <span class="chip">🚨 Crowd Safety Advice</span>
            <span class="chip">📦 Batch Processing</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Settings kept OUT of the sidebar (change the values here if you ever need to)
# ----------------------------------------------------------------------------
flip_tta = True                       # average prediction with the mirrored image
relu_density = True                   # remove negative density values
count_source = "Count head output"    # or "Density map sum" / "Average of both"
grid = 1                              # tiles per side (1 = whole image)
norm_csr, norm_mcnn, norm_hyb = NORM_MODES[1], NORM_MODES[1], NORM_MODES[1]   # preprocessing per model (0-1 for all; CSRNet gave counts ~3-4x too big with ImageNet)
f_csr = f_mcnn = f_hyb = 1.0          # calibration factors
hyb_out, hyb_res = "auto", "csrnet"   # hybrid output mode / fusion resolution
smooth_csr, smooth = 8, 4             # heatmap smoothing (CSRNet / others)
zone_n = 3                            # zone grid for the safety map
show_gallery = True
area_m2 = 1.0                         # levels use people counts, no m2 needed

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="logo">👥</div><div>'
        '<div class="name">Crowd Counting</div>'
        '<div class="tag">Density estimation </div></div></div>',
        unsafe_allow_html=True,
    )
    st.header("🧠 Model")
    model_choice = st.selectbox("Model", list(MODELS.keys()), index=1)
    st.caption("✅ MCNN is the recommended model. CSRNet and Hybrid are experimental. "
               "Use **Model comparison** on the page to compare all three.")

    st.header("🚨 Crowd safety")
    high_limit = st.number_input("High-crowd limit (people in the image)", 10, 100000, 200, 10,
                                 help="Number of people at which you treat the scene as HIGH crowd. "
                                      "Levels are set from this: Normal, Moderate, High, Severe, Critical.")
    alert_T = float(high_limit)

    st.header("🔥 Heatmap")
    alpha = st.slider("Opacity", 0.1, 0.9, 0.55, 0.05)
    st.markdown(f'<span class="device-badge">⚙️ Device: {DEVICE.type.upper()}</span>',
                unsafe_allow_html=True)
    st.markdown('<div class="side-foot">CSRNet · MCNN · Hybrid · PyTorch · Streamlit</div>',
                unsafe_allow_html=True)

NORM_BY = {"CSRNet": norm_csr, "MCNN": norm_mcnn, "Hybrid": norm_hyb}
FACTOR_BY = {"CSRNet": f_csr, "MCNN": f_mcnn, "Hybrid": f_hyb}
SMOOTH_BY = {"CSRNet": smooth_csr, "MCNN": smooth, "Hybrid": smooth}

with st.expander("🧪 Inspect a model file (layer names & shapes)", expanded=False):
    st.write("Shows what is inside a .pth file. Useful to check that the architecture in the code "
             "matches the saved weights.")
    insp = st.selectbox("Model file", [v[1] for v in MODELS.values()], key="insp_file")
    if st.button("Inspect file", key="insp_btn"):
        try:
            info, dfk = inspect_checkpoint(insp)
            if info:
                st.write("Top-level contents:", info)
            st.write(f"{len(dfk)} tensors")
            st.dataframe(dfk, use_container_width=True, hide_index=True)
            st.code("\n".join(f"{r['layer name']}  {r['shape']}" for r in dfk.to_dict("records")))
        except Exception as e:
            st.error(f"Could not read file: {e}")

section("📤 Upload images")
with st.container(border=True):
    files = st.file_uploader("Drag & drop or browse (select multiple)",
                             type=["jpg", "jpeg", "png", "bmp", "webp"],
                             accept_multiple_files=True)
if not files:
    st.markdown(
        """
        <div class="empty-card">
            <div class="big">📷</div>
            <h3>No images yet</h3>
            <p>Upload one or more crowd images to begin counting.</p>
        </div>
        <div class="steps">
            <div class="step"><div class="num">1</div><h4>Upload</h4>
                <p>Drop JPG, PNG, BMP or WEBP crowd photos.</p></div>
            <div class="step"><div class="num">2</div><h4>Configure</h4>
                <p>Pick a model and tune accuracy in the sidebar.</p></div>
            <div class="step"><div class="num">3</div><h4>Analyse</h4>
                <p>Get counts, heatmaps, CSV and ZIP downloads.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

model_names = [model_choice]
try:
    loaded = {m: load_model(m, CODE_STAMP) for m in model_names}
    for _m, (_mod, _sz) in loaded.items():
        if hasattr(_mod, "fuse_to"):
            _mod.fuse_to = hyb_res
            _mod.out_mode = hyb_out
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


def open_img(f):
    return Image.open(io.BytesIO(f.getvalue()))


with st.expander("🩺 Diagnose wrong count (test ONE image)", expanded=False):
    st.write("Pick an image whose real count you know. This tries every preprocessing / ReLU "
             "combination and shows both count outputs, so you can see which one matches.")
    pick = st.selectbox("Image", [f.name for f in files], key="diag_img")
    true_n = st.number_input("Real count", min_value=0, value=0, step=1, key="diag_true")
    if st.button("Run diagnostics"):
        f = next(x for x in files if x.name == pick)
        image = open_img(f)
        out = []
        for m in model_names:
            model, size = loaded[m]
            for nm in NORM_MODES:
                for rl in (True, False):
                    _, _, cs, ch = infer(model, size, image, 1, nm, False, rl)
                    out.append({"model": short_name(m), "preprocessing": nm, "ReLU": rl,
                                "density sum": round(cs, 1), "count head": round(ch, 1)})
        dd = pd.DataFrame(out)
        if true_n > 0:
            dd["density err %"] = ((dd["density sum"] - true_n).abs() / true_n * 100).round(0)
            dd["head err %"] = ((dd["count head"] - true_n).abs() / true_n * 100).round(0)
        st.dataframe(dd, use_container_width=True, hide_index=True)
        st.caption("Pick the row (and column) closest to the real count, set those options in the "
                   "settings block near the top of app.py. If one column is consistently ~N times too big, "
                   "use 1/N as the calibration factor there.")

    if any(hasattr(loaded[m][0], "debug") for m in model_names):
        st.markdown("---")
        st.write("**Hybrid internals:** what each part of the hybrid outputs for every preprocessing. "
                 "A healthy setting has MCNN / CSRNet sums close to the real count and a fusion output "
                 "(after ReLU) that is not zero.")
        if st.button("Show hybrid internals"):
            f = next(x for x in files if x.name == pick)
            wimg = _work_image(open_img(f))
            for m in model_names:
                mod_, size_ = loaded[m]
                if not hasattr(mod_, "debug"):
                    continue
                rows_dbg = []
                for nm in NORM_MODES:
                    x_ = tile_tensor(wimg, size_, nm).unsqueeze(0).to(DEVICE)
                    r_ = mod_.debug(x_)
                    rows_dbg.append({"preprocessing": nm, **{k: round(v, 2) for k, v in r_.items()}})
                st.dataframe(pd.DataFrame(rows_dbg), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.write("**See it, don't guess:** shows the heatmap for every preprocessing option. "
             "The correct one gives a clean heatmap on the real crowd and a count close to the truth.")
    if st.button("Show heatmaps for every preprocessing"):
        f = next(x for x in files if x.name == pick)
        image = open_img(f)
        for m in model_names:
            model, size = loaded[m]
            sn = short_name(m)
            st.markdown(f"**{sn}**")
            cols = st.columns(len(NORM_MODES))
            for col, nm in zip(cols, NORM_MODES):
                img_, dens_, cs_, ch_ = infer(model, size, image, 1, nm, False, True)
                _, ov_ = render_heatmap(img_, dens_, alpha, SMOOTH_BY[sn])
                col.image(ov_, caption=f"{nm} | density {cs_:.0f} | head {ch_:.0f}",
                          use_container_width=True)

st.write("")
if st.button(f"🚀 Process {len(files)} image(s)", type="primary", use_container_width=True):
    rows, outputs, analysis = [], [], {}
    bar = st.progress(0.0, text="Starting...")
    for i, f in enumerate(files):
        try:
            image = open_img(f)
            row = {"image": f.name}
            finals = {}
            for m in model_names:
                model, size = loaded[m]
                sn = short_name(m)
                img, dens, c_sum, c_head = infer(model, size, image, grid, NORM_BY[sn], flip_tta, relu_density)
                raw = pick_count(c_sum, c_head, count_source)
                final = raw * FACTOR_BY[sn]
                finals[sn] = final
                row[f"{sn}_count"] = round(final, 1)
                row[f"{sn}_raw"] = round(raw, 1)
                row[f"{sn}_density_sum"] = round(c_sum, 1)
                row[f"{sn}_count_head"] = round(c_head, 1)
                if hasattr(model, "last_used"):
                    row[f"{sn}_density_source"] = model.last_used
                heat, overlay = render_heatmap(img, dens, alpha, SMOOTH_BY[sn])
                outputs.append((f.name, sn, img, heat, overlay, final))
                analysis[(f.name, sn)] = coarse_sum(dens, 12)
            rows.append(row)
        except Exception as e:
            rows.append({"image": f.name, "error": str(e)})
        bar.progress((i + 1) / len(files), text=f"Processed {i + 1}/{len(files)}: {f.name}")
    bar.empty()
    st.session_state["rows"], st.session_state["outputs"] = rows, outputs
    st.session_state["analysis"] = analysis
    st.session_state.pop("xai", None)

if "rows" in st.session_state:
    df = pd.DataFrame(st.session_state["rows"])

    main_col = next((c for c in df.columns if c.endswith("_count")), None)

    section("📊 Summary")
    if main_col:
        c1, c2, c3 = st.columns(3)
        with c1:
            stat_card("🖼️", "Images processed", f"{len(df)}")
        with c2:
            stat_card("👥", f"Total people ({main_col})", f"{df[main_col].sum():.0f}")
        with c3:
            stat_card("📈", "Average per image", f"{df[main_col].mean():.1f}")

    section("📋 Results")
    with st.container(border=True):
        st.dataframe(df, use_container_width=True)
        if main_col and main_col in df.columns and len(df) > 1:
            st.caption("People counted per image")
            st.bar_chart(df.dropna(subset=[main_col]).set_index("image")[main_col],
                         color="#6C63FF")

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download CSV report", df.to_csv(index=False).encode("utf-8"),
                       "crowd_counts.csv", "text/csv", use_container_width=True)
    outputs = st.session_state["outputs"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, sn, _, heat, overlay, final in outputs:
            stem = os.path.splitext(name)[0]
            for tag, im in (("overlay", overlay), ("density", heat)):
                b = io.BytesIO()
                im.save(b, format="PNG")
                z.writestr(f"{stem}_{sn}_{tag}.png", b.getvalue())
    d2.download_button("⬇️ Download heatmaps (ZIP)", buf.getvalue(),
                       "crowd_heatmaps.zip", "application/zip", use_container_width=True)

    # ------------------------------------------------------------------
    # Crowd safety & recommendation system
    # ------------------------------------------------------------------
    analysis = st.session_state.get("analysis", {})
    ok_rows = [r for r in st.session_state["rows"] if "error" not in r]
    if ok_rows:
        used = [sn for sn in ("CSRNet", "MCNN", "Hybrid") if f"{sn}_count" in ok_rows[0]]
        total_key = f"{used[0]}_count"
        zone_area = area_m2 / (zone_n ** 2)

        def cells_for(r):
            """12x12 grid of people counts, scaled to the image's final count."""
            parts = {}
            for sn_ in used:
                co_ = analysis.get((r["image"], sn_))
                if co_ is None or co_.sum() <= 1e-6:
                    return None
                parts[sn_] = co_ / co_.sum() * r[f"{sn_}_count"]
            cells = parts[used[0]]
            if cells.sum() <= 1e-6:   # model produced an empty density map: zones unreliable
                return None
            return cells / cells.sum() * float(r[total_key])

        def assess(r):
            total = float(r[total_key])
            dens = total / area_m2
            cells = cells_for(r)
            zc = zdens = None
            peak = dens
            if cells is not None:
                zc = to_zones(cells, zone_n)
                zdens = zc / zone_area
                peak = float(zdens.max())
            lvl = max(0, risk_level(dens, alert_T), risk_level(peak, alert_T) - 1)
            hot = []
            if zdens is not None:
                hot = sorted(((zone_name(zr, zcn, zone_n), float(zc[zr, zcn]))
                              for (zr, zcn), v in np.ndenumerate(zdens) if v >= alert_T),
                             key=lambda t: -t[1])
            peak_people = float(zc.max()) if zc is not None else total
            return dict(total=total, dens=dens, zc=zc, zdens=zdens, peak=peak, lvl=lvl, hot=hot,
                        peak_people=peak_people)

        assessed = {r["image"]: assess(r) for r in ok_rows}

        section("🚨 Crowd safety & recommendations")
        st.markdown('<div class="section-sub">Turns the count into a risk level and tells you what to do. '
                    'Set the high-crowd limit in the sidebar.</div>', unsafe_allow_html=True)

        T = alert_T
        ranges = [f"&lt; {0.5 * T:.0f}", f"{0.5 * T:.0f} - {T:.0f}", f"{T:.0f} - {2 * T:.0f}",
                  f"{2 * T:.0f} - {2.75 * T:.0f}", f"&ge; {2.75 * T:.0f}"]
        st.markdown('<div class="legend">' + "".join(
            f'<span class="lg" style="background:{col}">{ic} {nm}: {rg} people</span>'
            for (nm, ic, col), rg in zip(LEVELS, ranges)) + '</div>', unsafe_allow_html=True)

        n_high = sum(1 for a_ in assessed.values() if a_["lvl"] >= 2)
        k1, k2, k3 = st.columns(3)
        with k1:
            stat_card("🚩", "Images at HIGH risk or above", f"{n_high} / {len(assessed)}")
        with k2:
            stat_card("👥", "Most people in one image", f"{max(a_['total'] for a_ in assessed.values()):.0f}")
        with k3:
            stat_card("🔥", "Busiest zone", f"{max(a_['peak_people'] for a_ in assessed.values()):.0f} people")

        risk_df = pd.DataFrame([{
            "image": n_, "people": round(a_["total"], 1),
            "busiest zone (people)": round(a_["peak_people"], 0),
            "risk level": f"{LEVELS[a_['lvl']][1]} {LEVELS[a_['lvl']][0]}",
            "hotspot zones": len(a_["hot"]),
        } for n_, a_ in assessed.items()])
        st.write("")
        with st.container(border=True):
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download safety report (CSV)",
                               risk_df.to_csv(index=False).encode("utf-8"),
                               "crowd_safety_report.csv", "text/csv")

        rname = st.selectbox("Select an image for detailed advice", list(assessed.keys()), key="rec_img")
        a = assessed[rname]
        lname, licon, lcolor = LEVELS[a["lvl"]]
        st.markdown(
            f'<div class="risk-banner" style="--rc:{lcolor}"><div class="rb-icon">{licon}</div><div>'
            f'<div class="rb-title">{lname.upper()} crowd level</div>'
            f'<div class="rb-sub">{a["total"]:.0f} people &middot; busiest zone {a["peak_people"]:.0f} people '
            f'&middot; {len(a["hot"])} hotspot zone(s)</div>'
            f'</div></div>', unsafe_allow_html=True)

        if a["zc"] is None:
            st.warning("The model's density map is empty for this image, so the zone map is not available "
                       "(the total count still comes from the count head). Open Diagnose > 'Hybrid internals' "
                       "to see what each part of the model outputs.")
        if a["zc"] is not None:
            base_img = next(t[2] for t in outputs if t[0] == rname)
            left, right = st.columns([3, 2])
            with left:
                st.image(zone_overlay(base_img, a["zc"], area_m2, alert_T),
                         caption=f"Zone map ({zone_n}x{zone_n}): colour = crowd level, text = people and level",
                         use_container_width=True)
            with right:
                st.markdown("**People per zone**")
                st.dataframe(pd.DataFrame(a["zc"].round(0).astype(int),
                                          index=[f"Row {i + 1}" for i in range(zone_n)],
                                          columns=[f"Col {j + 1}" for j in range(zone_n)]),
                             use_container_width=True)
                st.markdown("**Crowd level per zone**")
                st.dataframe(pd.DataFrame(np.vectorize(lambda v: LEVELS[risk_level(v, alert_T)][0], otypes=[object])(a["zdens"]),
                                          index=[f"Row {i + 1}" for i in range(zone_n)],
                                          columns=[f"Col {j + 1}" for j in range(zone_n)]),
                             use_container_width=True)

        extra = []
        if a["hot"]:
            extra.append(("📍", "Hotspot zones (densest first): " + ", ".join(
                f"<b>{n_}</b> ({v:.0f} people)" for n_, v in a["hot"][:5])))
        if a["zdens"] is not None and a["lvl"] >= 2:
            order = sorted(((zone_name(zr, zcn, zone_n), float(v))
                            for (zr, zcn), v in np.ndenumerate(a["zc"])), key=lambda t: t[1])
            extra.append(("🟢", "Quietest zones - use them to redirect people: " + ", ".join(
                f"<b>{n_}</b> ({v:.0f} people)" for n_, v in order[:2])))
        excess = a["total"] - alert_T
        if excess > 0:
            extra.append(("⚖️", f"About <b>{excess:.0f}</b> people above the high-crowd limit - "
                                f"aim to bring the area down to about <b>{alert_T:.0f}</b> people."))

        st.markdown("#### 📋 Safety suggestions")
        items = extra + RECS[a["lvl"]]
        st.markdown('<div class="rec-grid">' + "".join(
            f'<div class="rec" style="--rc:{lcolor}"><div class="ri">{ic}</div><div class="rt">{tx}</div></div>'
            for ic, tx in items) + '</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="disclaimer">Decision-support only. Zones are equal-sized, so camera perspective can '
            'over- or under-state how crowded a zone really is. Level = the higher of the whole-scene level and '
            'one step below the busiest zone. Always follow your venue\'s safety plan and on-site officials.</div>', unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # Explainable AI
        # ------------------------------------------------------------------
        section("🔍 Explainable AI (XAI)")
        st.markdown('<div class="section-sub">See <i>why</i> the model gave this count: where it looked, '
                    'and where the people are.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            xc1, xc2 = st.columns(2)
            xname = xc1.selectbox("Image", [r["image"] for r in ok_rows], key="xai_img")
            xsn = xc2.selectbox("Model to explain", used, key="xai_model")
            xrow = next(r for r in ok_rows if r["image"] == xname)
            xfile = next((f for f in files if f.name == xname), None)
            if st.button("🔍 Generate explanation", key="xai_btn"):
                if xfile is None:
                    st.warning("Upload this image again to explain it.")
                else:
                    m_full = next(m for m in MODELS if short_name(m) == xsn)
                    model_, size_ = load_model(m_full, CODE_STAMP)
                    try:
                        with st.spinner("Computing Grad-CAM..."):
                            gimg, gcam = gradcam(model_, size_, open_img(xfile), NORM_BY[xsn])
                        st.session_state["xai"] = {"key": (xname, xsn), "img": gimg, "cam": gcam}
                    except Exception as e:
                        st.error(f"Could not compute Grad-CAM: {e}")

            xai = st.session_state.get("xai")
            if xai and xai["key"] == (xname, xsn):
                _, cam_ov = render_heatmap(xai["img"], xai["cam"], max(alpha, 0.6), 3)
                dens_heat = next(t[3] for t in outputs if t[0] == xname and t[1] == xsn)
                dens_ov = next(t[4] for t in outputs if t[0] == xname and t[1] == xsn)
                g1, g2, g3 = st.columns(3)
                g1.image(xai["img"], caption="Original", use_container_width=True)
                g2.image(cam_ov, caption="Grad-CAM: where the model looked", use_container_width=True)
                g3.image(dens_ov, caption="Predicted density heatmap", use_container_width=True)
            else:
                st.info("Click **Generate explanation** to see the Grad-CAM attention map.")

            co = analysis.get((xname, xsn))
            if co is not None:
                share = co / max(co.sum(), 1e-8)
                pts = explain_text(xsn, float(xrow[f"{xsn}_count"]), share)
                st.markdown('<div class="xai-card"><h4>💬 Explanation </h4><ul>' +
                            "".join(f"<li>{p}</li>" for p in pts) + '</ul></div>', unsafe_allow_html=True)

                st.markdown("**Where does the count come from?** (share of all people in each zone)")
                z_share = to_zones(share, zone_n) * 100
                zdf = pd.DataFrame(z_share, index=[f"Row {i + 1}" for i in range(zone_n)],
                                   columns=[f"Col {j + 1}" for j in range(zone_n)])
                st.dataframe(zdf.style.format("{:.1f}%").background_gradient(cmap="Purples"),
                             use_container_width=True)


    # ------------------------------------------------------------------
    # Accuracy check: enter true counts -> error metrics, factor, best settings
    # ------------------------------------------------------------------
    section("🎯 Accuracy & calibration")
    with st.expander("Enter known counts to measure accuracy", expanded=False):
        st.write("Type the **real count** for any images you know (e.g. from the "
                 "ShanghaiTech ground truth). Use 10+ images for a trustworthy result.")
        base = pd.DataFrame({"image": [r["image"] for r in st.session_state["rows"]],
                             "true_count": np.nan})
        edited = st.data_editor(base, key="truth_editor", use_container_width=True,
                                disabled=["image"], hide_index=True)
        truth = {r.image: r.true_count for r in edited.itertuples() if pd.notna(r.true_count)}

        if truth:
            for m in model_names:
                sn = short_name(m)
                sub = df[df["image"].isin(truth) & df[f"{sn}_raw"].notna()]
                if sub.empty:
                    continue
                t = sub["image"].map(truth).astype(float).values
                p_final = sub[f"{sn}_count"].values.astype(float)
                p_raw = sub[f"{sn}_raw"].values.astype(float)
                mae = np.abs(t - p_final).mean()
                rmse = np.sqrt(((t - p_final) ** 2).mean())
                mape = (np.abs(t - p_final) / np.maximum(t, 1)).mean() * 100
                suggested = t.sum() / max(p_raw.sum(), 1e-8)
                st.markdown(f"**{sn}** - MAE `{mae:.1f}` · RMSE `{rmse:.1f}` · "
                            f"MAPE `{mape:.1f}%` · suggested calibration factor "
                            f"`{suggested:.3f}`")

            if st.button("🔍 Find best settings (tiles x flip x count source)"):
                rows_sw = []
                prog = st.progress(0.0, text="Testing settings...")
                names = [f.name for f in files if f.name in truth]
                combos = [(m, g, fl) for m in model_names for g in (1, 2, 3, 4) for fl in (False, True)]
                for ci, (m, g, fl) in enumerate(combos):
                    model, size = loaded[m]
                    preds = {s: [] for s in COUNT_SOURCES}
                    for f in files:
                        if f.name not in truth:
                            continue
                        _, _, cs, ch = infer(model, size, open_img(f), g, NORM_BY[short_name(m)], fl, relu_density)
                        for s in COUNT_SOURCES:
                            preds[s].append(pick_count(cs, ch, s))
                    t = np.array([truth[n] for n in names], dtype=float)
                    for s in COUNT_SOURCES:
                        p = np.array(preds[s])
                        k = t.sum() / max(p.sum(), 1e-8)
                        rows_sw.append({"model": short_name(m), "tiles": f"{g}x{g}",
                                        "flip": fl, "source": s,
                                        "MAE (raw)": round(np.abs(t - p).mean(), 1),
                                        "best factor": round(k, 3),
                                        "MAE (after factor)": round(np.abs(t - p * k).mean(), 1)})
                    prog.progress((ci + 1) / len(combos), text=f"Tested {ci + 1}/{len(combos)}")
                prog.empty()
                st.session_state["sweep"] = pd.DataFrame(rows_sw).sort_values("MAE (after factor)")

            if "sweep" in st.session_state:
                st.write("Best settings first. Update the settings block near the top of app.py, then re-process:")
                st.dataframe(st.session_state["sweep"], use_container_width=True, hide_index=True)

    if show_gallery:
        section("🖼️ Gallery")
        for gi, (name, sn, img, heat, overlay, final) in enumerate(outputs):
            with st.expander(f"{name} · {sn} · {final:.0f} people", expanded=(gi == 0)):
                st.markdown(f'<div class="count-badge">👥 {final:.0f} people</div>',
                            unsafe_allow_html=True)
                a, b, c = st.columns(3)
                a.image(img, caption="Original", use_container_width=True)
                b.image(heat, caption="Density map", use_container_width=True)
                c.image(overlay, caption="Overlay", use_container_width=True)


# ----------------------------------------------------------------------------
# Model comparison (kept at the very end of the page)
# ----------------------------------------------------------------------------
section("⚖️ Model comparison")
with st.container(border=True):
    st.write("Run **all three models** (CSRNet, MCNN, Hybrid) on the same image and compare their "
             "counts and heatmaps side by side.")
    cc1, cc2 = st.columns([3, 1])
    cmp_img = cc1.selectbox("Image to compare", [f.name for f in files], key="cmp_img")
    cmp_true = cc2.number_input("Real count (optional)", min_value=0, value=0, step=1, key="cmp_true")
    if st.button("⚖️ Compare all models", key="cmp_btn"):
        f_cmp = next(x for x in files if x.name == cmp_img)
        image_cmp = open_img(f_cmp)
        res_cmp = []
        with st.spinner("Running all models..."):
            for m_ in MODELS:
                sn_ = short_name(m_)
                try:
                    model_c, size_c = load_model(m_, CODE_STAMP)
                    if hasattr(model_c, "fuse_to"):
                        model_c.fuse_to, model_c.out_mode = hyb_res, hyb_out
                    img_c, dens_c, cs_c, ch_c = infer(model_c, size_c, image_cmp, grid, NORM_BY[sn_],
                                                      flip_tta, relu_density)
                    cnt_c = pick_count(cs_c, ch_c, count_source) * FACTOR_BY[sn_]
                    _, ov_c = render_heatmap(img_c, dens_c, alpha, SMOOTH_BY[sn_])
                    res_cmp.append({"model": sn_, "count": cnt_c, "density sum": cs_c,
                                    "count head": ch_c, "overlay": ov_c})
                except Exception as e:
                    res_cmp.append({"model": sn_, "error": str(e)})
        st.session_state["cmp"] = {"name": cmp_img, "true": cmp_true, "res": res_cmp}

    cmp = st.session_state.get("cmp")
    if cmp and cmp["name"] == cmp_img:
        good = [r for r in cmp["res"] if "error" not in r]
        true_c = cmp["true"]
        best_model = min(good, key=lambda r: abs(r["count"] - true_c))["model"] if (good and true_c > 0) else None
        cols_c = st.columns(len(cmp["res"]))
        for col_, r in zip(cols_c, cmp["res"]):
            with col_:
                if "error" in r:
                    st.error(f"{r['model']}: {r['error']}")
                    continue
                badge = " ✅ closest" if r["model"] == best_model else ""
                st.markdown(f"**{r['model']}{badge}**")
                st.image(r["overlay"], use_container_width=True)
                st.markdown(f'<div class="count-badge">👥 {r["count"]:.0f} people</div>', unsafe_allow_html=True)
        if good:
            rows_c = []
            for r in good:
                row_c = {"model": r["model"], "count": round(r["count"], 1),
                         "density sum": round(r["density sum"], 1), "count head": round(r["count head"], 1)}
                if true_c > 0:
                    row_c["error (people)"] = round(abs(r["count"] - true_c), 1)
                    row_c["error %"] = round(abs(r["count"] - true_c) / true_c * 100, 1)
                rows_c.append(row_c)
            dfc = pd.DataFrame(rows_c)
            st.dataframe(dfc, use_container_width=True, hide_index=True)
            if len(good) > 1:
                st.bar_chart(dfc.set_index("model")["count"], color="#6C63FF")
                counts_c = [r["count"] for r in good]
                st.caption(f"Models differ by up to {max(counts_c) - min(counts_c):.0f} people "
                           f"({(max(counts_c) - min(counts_c)) / max(np.mean(counts_c), 1) * 100:.0f}% of the average). "
                           + ("The closest model to your real count is marked with ✅." if best_model else
                              "Enter the real count above to see which model is closest."))