"""
2-D toy datasets. Each "image" x_0 is just a point (x, y), so the whole data
distribution can be drawn on one plot and we can literally watch DDPM work.
"""

import math

import numpy as np
import torch


def text_points(n, text="DDPM"):
    """Points sampled uniformly inside the letters of `text`."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path
    from matplotlib.textpath import TextPath

    glyphs = TextPath((0, 0), text, size=1, prop=FontProperties(family="DejaVu Sans", weight="bold"))
    polygons = [Path(p) for p in glyphs.to_polygons()]
    box = glyphs.get_extents()

    points = []
    while sum(len(p) for p in points) < n:
        cand = np.random.rand(4 * n, 2) * [box.width, box.height] + [box.x0, box.y0]
        # Even-odd rule: inside an odd number of outlines = filled (handles the holes in D, P, ...)
        inside = sum(poly.contains_points(cand).astype(int) for poly in polygons) % 2 == 1
        points.append(cand[inside])
    return np.concatenate(points)[:n]


def swiss_roll(n):
    theta = 1.5 * math.pi * (1 + 2 * np.random.rand(n))
    pts = np.stack([theta * np.cos(theta), theta * np.sin(theta)], axis=1)
    return pts + 0.4 * np.random.randn(n, 2)


def two_moons(n):
    n1 = n // 2
    a = np.random.rand(n1) * math.pi
    b = np.random.rand(n - n1) * math.pi
    upper = np.stack([np.cos(a), np.sin(a)], axis=1)
    lower = np.stack([1 - np.cos(b), 0.5 - np.sin(b)], axis=1)
    return np.concatenate([upper, lower]) + 0.06 * np.random.randn(n, 2)


def get_dataset(name, n=50_000, text="DDPM"):
    if name == "text":
        pts = text_points(n, text)
    elif name == "swissroll":
        pts = swiss_roll(n)
    elif name == "moons":
        pts = two_moons(n)
    else:
        raise ValueError(f"unknown dataset: {name}")

    # Centre the data and scale it into [-2.5, 2.5], about the same spread as the N(0, I) noise it is mixed with.
    pts = pts - pts.mean(axis=0)
    pts = 2.5 * pts / np.abs(pts).max()
    return torch.tensor(pts, dtype=torch.float32)
