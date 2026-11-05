#!/usr/bin/env python3
"""Rebuild Figures 1 and 2 of the paper from the images saved by secure_medimg_experiments.py.

Usage:  python make_figures.py --results results_native --out figures
Figure 1: original / AES-128-GCM ciphertext shown as an image / decrypted image.
Figure 2: 5% random pixel loss (top) and 5% lost CBC ciphertext blocks (bottom):
          original, corrupted, TELEA, Navier-Stokes (PSNR/SSIM of this single image in the titles).
"""
import argparse, os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from skimage.metrics import structural_similarity as ssim

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results_native")
ap.add_argument("--out", default="figures")
a = ap.parse_args()
F = os.path.join(a.results, "figs")
os.makedirs(a.out, exist_ok=True)

orig = cv2.imread(os.path.join(F, "original.png"), 0)
h, w = orig.shape
key, nonce = get_random_bytes(16), get_random_bytes(12)
ct, tag = AES.new(key, AES.MODE_GCM, nonce=nonce).encrypt_and_digest(orig.tobytes())
enc = __import__("numpy").frombuffer(ct, "uint8").reshape(h, w)
dec = __import__("numpy").frombuffer(
    AES.new(key, AES.MODE_GCM, nonce=nonce).decrypt_and_verify(ct, tag), "uint8").reshape(h, w)
assert (dec == orig).all()

fig, ax = plt.subplots(1, 3, figsize=(9, 2.6))
for axis, im, t in zip(ax, [orig, enc, dec], ["Original", "Encrypted (AES-128-GCM)", "Decrypted (bit-identical)"]):
    axis.imshow(im, cmap="gray", vmin=0, vmax=255); axis.set_title(t, fontsize=9); axis.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig1_pipeline.png"), dpi=300); plt.close()

def load(n): return cv2.imread(os.path.join(F, n + ".png"), 0)
rows = [("5% random pixel loss", "randompixel"), ("5% of CBC ciphertext blocks lost", "ciphertextloss")]
fig, ax = plt.subplots(2, 4, figsize=(10, 4.4))
for r, (lab, tg) in enumerate(rows):
    panels = [("Original", orig), ("Corrupted", load(tg + "_corrupted")), ("TELEA", load(tg + "_telea")),
              ("Navier-Stokes", load(tg + "_ns"))]
    for c, (t, im) in enumerate(panels):
        title = t if t == "Original" else f"{t}\nPSNR {cv2.PSNR(orig, im):.1f} dB, SSIM {ssim(orig, im, data_range=255):.3f}"
        ax[r, c].imshow(im, cmap="gray", vmin=0, vmax=255); ax[r, c].set_title(title, fontsize=8); ax[r, c].axis("off")
    ax[r, 0].text(-0.05, 0.5, lab, transform=ax[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig2_recovery.png"), dpi=300); plt.close()
print("saved to", a.out)
