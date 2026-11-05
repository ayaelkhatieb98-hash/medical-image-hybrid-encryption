#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
secure_medimg_experiments.py
============================
Re-run of the experiments of "Enhancing Medical Image Reconstruction Using
RSA-OAEP and Hybrid RSA-AES Encryption", written to answer the reviewers.

Reviewer comment -> what this script does
-----------------------------------------
R2-C1 / R1-2  Pure-RSA timing   : (a) pixel-wise textbook RSA measured on a sample and extrapolated
                                  with the arithmetic printed (seconds, no unit mix-up), optional
                                  DIRECT full-image encryption (--rsa_pixel_full_enc);
                                  (b) chunked RSA-OAEP of the whole image, measured DIRECTLY
                                  (a fairer "pure RSA" baseline than pixel-wise textbook RSA).
R2-C2 / R1-3  SHA-256 vs AEAD   : AES-CBC+SHA-256(plaintext) [original] vs AES-CBC+HMAC-SHA256
                                  (Encrypt-then-MAC) vs AES-GCM vs ChaCha20-Poly1305, plus
                                  tamper / forgery tests (+ optional RSA-PSS sender signature).
R2-C4 / R1-1  Single image      : runs on every image found in --data_dir (up to --max_images).
R2-C5         Timing stats      : --repeats runs per image/scheme, mean, SD, 95% CI, median.
R2-C6         NPCR/UACI, key    : NPCR/UACI with FIXED key+IV and a 1-bit plaintext flip (the
                                  meaningful diffusion test), and with fresh keys (randomness);
                                  key sensitivity decrypts WITHOUT unpadding (no black fallback).
R2-C8         Modern AEAD       : AES-128/256-GCM, ChaCha20-Poly1305, CBC+HMAC, mean +/- SD.
R1-6          Corruption models : random pixel loss (1/5/10/20 %), block loss, burst loss and
                                  ciphertext-domain CBC block loss; TELEA vs Navier-Stokes vs a
                                  simple neighbour-mean baseline.
R1-7          Reproducibility   : seeds, library versions, hardware, file SHA-256 of every image
                                  and all raw results are written to --out_dir.

Usage
-----
  python secure_medimg_experiments.py --data_dir /path/to/kaggle_knee_images --out_dir results
  python secure_medimg_experiments.py --data_dir imgs --resize 415x720 --max_images 100 --repeats 30
  python secure_medimg_experiments.py --synthetic 3 --quick          # smoke test only (NOT for the paper)
  python secure_medimg_experiments.py --data_dir imgs --skip rsa     # skip the slow RSA baselines

Notes
-----
* Key material and IVs for real encryptions come from the OS CSPRNG (never seeded). Only the numpy
  RNG (image selection, masks, bit positions, sample positions) is seeded via --seed.
* Fixed key/IV (and GCM nonce reuse) is used ONLY inside the NPCR/UACI diffusion test.
"""
import argparse
import gc
import hashlib
import hmac as hmac_std
import json
import os
import platform
import struct
import sys
import time

import cv2
import numpy as np
import pandas as pd
import skimage
import scipy
from scipy import stats
from skimage.metrics import structural_similarity as _ssim
from Crypto.Cipher import AES, ChaCha20_Poly1305, PKCS1_OAEP
from Crypto.Hash import HMAC, SHA256
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Signature import pss
from Crypto.Util.Padding import pad, unpad
import Crypto

NPCR_IDEAL, UACI_IDEAL = 99.6094, 33.4635   # ideal values for 8-bit images

# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def entropy_bytes(arr):
    hist = np.bincount(np.asarray(arr, np.uint8).ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def chi2_uniform(arr):
    hist = np.bincount(np.asarray(arr, np.uint8).ravel(), minlength=256)
    chi2, p = stats.chisquare(hist)
    return float(chi2), float(p)


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a, b):
    m = mse(a, b)
    return float("inf") if m == 0 else float(10 * np.log10(255.0 ** 2 / m))


def ssim(a, b):
    return float(_ssim(a, b, data_range=255))


def adjacent_corr(img, rng, n_pairs=5000):
    h, w = img.shape
    out = {}
    for name, (dy, dx) in {"H": (0, 1), "V": (1, 0), "D": (1, 1)}.items():
        ys = rng.integers(0, h - dy, n_pairs)
        xs = rng.integers(0, w - dx, n_pairs)
        a = img[ys, xs].astype(np.float64)
        b = img[ys + dy, xs + dx].astype(np.float64)
        out[name] = float("nan") if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
    return out


def bytes_as_image(buf, shape):
    h, w = shape
    return np.frombuffer(buf, np.uint8)[: h * w].reshape(h, w)


def npcr_uaci(c1, c2):
    a = np.frombuffer(c1, np.uint8)
    b = np.frombuffer(c2, np.uint8)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    npcr = 100.0 * float(np.mean(a != b))
    uaci = 100.0 * float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)) / 255.0))
    return npcr, uaci


def pm(x, d=3):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "n/a"
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return f"{x.mean():.{d}f} ± {sd:.{d}f}"


def ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean() - h), float(x.mean() + h)


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Schemes (symmetric part). The hybrid layer below wraps the key material with RSA-2048-OAEP.
# ----------------------------------------------------------------------------------------------
class Scheme:
    label = ""
    km_len = 16      # bytes of key material wrapped by RSA-OAEP
    iv_len = 16
    authenticated = False

    def encrypt_sym(self, km, iv, data, aad):   # -> (ciphertext, tag)
        raise NotImplementedError

    def decrypt_sym(self, km, iv, ct, tag, aad):  # -> plaintext, raises ValueError on failure
        raise NotImplementedError


class CBCLegacy(Scheme):
    """Original paper: AES-128-CBC; SHA-256 of the PLAINTEXT is sent next to the ciphertext."""
    label = "CBC+SHA256 (original)"

    def encrypt_sym(self, km, iv, data, aad):
        ct = AES.new(km[:16], AES.MODE_CBC, iv).encrypt(pad(data, 16))
        return ct, hashlib.sha256(data).digest()

    def decrypt_sym(self, km, iv, ct, tag, aad):
        data = unpad(AES.new(km[:16], AES.MODE_CBC, iv).decrypt(ct), 16)
        if not hmac_std.compare_digest(hashlib.sha256(data).digest(), tag):
            raise ValueError("SHA-256 mismatch")
        return data


class CBCHMAC(Scheme):
    """AES-128-CBC + HMAC-SHA256, Encrypt-then-MAC with independent keys; header (shape) is MACed."""
    label = "CBC+HMAC-SHA256"
    km_len = 48
    authenticated = True

    def encrypt_sym(self, km, iv, data, aad):
        ct = AES.new(km[:16], AES.MODE_CBC, iv).encrypt(pad(data, 16))
        return ct, HMAC.new(km[16:48], aad + iv + ct, SHA256).digest()

    def decrypt_sym(self, km, iv, ct, tag, aad):
        HMAC.new(km[16:48], aad + iv + ct, SHA256).verify(tag)       # before any decryption
        return unpad(AES.new(km[:16], AES.MODE_CBC, iv).decrypt(ct), 16)


class GCM(Scheme):
    iv_len = 12
    authenticated = True

    def __init__(self, bits):
        self.klen = bits // 8
        self.km_len = self.klen
        self.label = f"AES-{bits}-GCM"

    def encrypt_sym(self, km, iv, data, aad):
        c = AES.new(km[: self.klen], AES.MODE_GCM, nonce=iv)
        c.update(aad)
        return c.encrypt_and_digest(data)

    def decrypt_sym(self, km, iv, ct, tag, aad):
        c = AES.new(km[: self.klen], AES.MODE_GCM, nonce=iv)
        c.update(aad)
        return c.decrypt_and_verify(ct, tag)


class ChaCha(Scheme):
    label = "ChaCha20-Poly1305"
    km_len = 32
    iv_len = 12
    authenticated = True

    def encrypt_sym(self, km, iv, data, aad):
        c = ChaCha20_Poly1305.new(key=km[:32], nonce=iv)
        c.update(aad)
        return c.encrypt_and_digest(data)

    def decrypt_sym(self, km, iv, ct, tag, aad):
        c = ChaCha20_Poly1305.new(key=km[:32], nonce=iv)
        c.update(aad)
        return c.decrypt_and_verify(ct, tag)


SCHEME_LEGACY, SCHEME_CBCH, SCHEME_GCM128, SCHEME_GCM256, SCHEME_CHACHA = (
    CBCLegacy(), CBCHMAC(), GCM(128), GCM(256), ChaCha())
CORE_SCHEMES = [SCHEME_LEGACY, SCHEME_CBCH, SCHEME_GCM128, SCHEME_GCM256, SCHEME_CHACHA]
# (label, scheme, with RSA-PSS sender signature)
CONFIGS = [(s.label, s, False) for s in CORE_SCHEMES]
CONFIGS.insert(2, (SCHEME_CBCH.label + " + RSA-PSS", SCHEME_CBCH, True))
CONFIGS.insert(4, (SCHEME_GCM128.label + " + RSA-PSS", SCHEME_GCM128, True))


class Hybrid:
    """RSA-2048-OAEP(SHA-256) key wrapping + symmetric layer (+ optional RSA-PSS sender signature)."""

    def __init__(self, recv_key, sender_key):
        self.recv_priv, self.recv_pub = recv_key, recv_key.publickey()
        self.sender_priv, self.sender_pub = sender_key, sender_key.publickey()
        self.enc_c = PKCS1_OAEP.new(self.recv_pub, hashAlgo=SHA256)
        self.dec_c = PKCS1_OAEP.new(self.recv_priv, hashAlgo=SHA256)

    @staticmethod
    def aad(shape):
        return struct.pack(">II", *shape)

    def encrypt(self, scheme, data, shape, sign=False, km=None, iv=None, wrap_with=None):
        km = km or get_random_bytes(scheme.km_len)
        iv = iv or get_random_bytes(scheme.iv_len)
        ct, tag = scheme.encrypt_sym(km, iv, data, self.aad(shape))
        pkg = dict(wk=(wrap_with or self.enc_c).encrypt(km), iv=iv, ct=ct, tag=tag, shape=tuple(shape), sig=None)
        if sign:
            pkg["sig"] = self.sign(pkg)
        return pkg

    def decrypt(self, scheme, pkg, verify_sig=False):
        if verify_sig:
            if not self.verify(pkg):
                raise ValueError("signature invalid")
        km = self.dec_c.decrypt(pkg["wk"])
        return scheme.decrypt_sym(km, pkg["iv"], pkg["ct"], pkg["tag"], self.aad(pkg["shape"]))

    @staticmethod
    def _blob(pkg):
        return b"".join([pkg["wk"], pkg["iv"], pkg["ct"], pkg["tag"], struct.pack(">II", *pkg["shape"])])

    def sign(self, pkg, key=None):
        return pss.new(key or self.sender_priv).sign(SHA256.new(self._blob(pkg)))

    def verify(self, pkg):
        if pkg.get("sig") is None:
            return False
        try:
            pss.new(self.sender_pub).verify(SHA256.new(self._blob(pkg)), pkg["sig"])
            return True
        except (ValueError, TypeError):
            return False


# ----------------------------------------------------------------------------------------------
# Data loading / environment
# ----------------------------------------------------------------------------------------------
def rng_for(seed, name):
    s = int.from_bytes(hashlib.sha256(f"{seed}-{name}".encode()).digest()[:8], "big")
    return np.random.default_rng(s)


def synthetic_images(n, rng, size=(256, 256)):
    """Smooth textured pseudo-radiographs for smoke tests ONLY (never report results on these)."""
    imgs = []
    for _ in range(n):
        h, w = size
        base = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 12)
        base = (base - base.min()) / (np.ptp(base) + 1e-9)
        canvas = (base * 90).astype(np.float32)
        for _ in range(4):
            c = (int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4)))
            ax = (int(rng.integers(w // 10, w // 4)), int(rng.integers(h // 8, h // 3)))
            tmp = np.zeros((h, w), np.float32)
            cv2.ellipse(tmp, c, ax, float(rng.integers(0, 180)), 0, 360, float(rng.integers(60, 140)), -1)
            canvas += cv2.GaussianBlur(tmp, (0, 0), 6)
        canvas += rng.normal(0, 3, (h, w))
        imgs.append(np.clip(canvas, 0, 255).astype(np.uint8))
    return imgs


def load_images(args, rng):
    items = []
    if args.synthetic:
        for i, im in enumerate(synthetic_images(args.synthetic, rng)):
            if args.resize:
                im = cv2.resize(im, (args.resize[1], args.resize[0]), interpolation=cv2.INTER_AREA)
            items.append(dict(id=f"synthetic_{i:03d}", path="synthetic", img=im,
                              file_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
        return items
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = []
    for root, _, names in os.walk(args.data_dir):
        files += [os.path.join(root, n) for n in names if n.lower().endswith(exts)]
    files = sorted(files)
    if not files:
        sys.exit(f"No images found under {args.data_dir}")
    if len(files) > args.max_images:
        files = [files[i] for i in np.sort(rng.choice(len(files), args.max_images, replace=False))]
    for p in files:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        if args.resize:
            h, w = args.resize
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        with open(p, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        items.append(dict(id=os.path.relpath(p, args.data_dir), path=p, img=im, file_sha256=sha))
    return items


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment(args, items):
    return dict(
        python=sys.version, platform=platform.platform(), cpu=cpu_model(), cpu_count=os.cpu_count(),
        versions=dict(pycryptodome=Crypto.__version__, opencv=cv2.__version__, numpy=np.__version__,
                      scikit_image=skimage.__version__, scipy=scipy.__version__, pandas=pd.__version__),
        args={k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
        numpy_seed=args.seed,
        images=[dict(id=i["id"], shape=list(i["img"].shape), file_sha256=i["file_sha256"]) for i in items],
    )


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------------------------
# Experiment 1: round trip, integrity, entropy, chi-square, adjacent-pixel correlation
# ----------------------------------------------------------------------------------------------
def exp_basic(items, hyb, args):
    rng = rng_for(args.seed, "basic")
    rows = []
    for it in items:
        img, data, shape = it["img"], it["img"].tobytes(), it["img"].shape
        co = adjacent_corr(img, rng)
        rows.append(dict(image=it["id"], scheme="ORIGINAL", ct_entropy=entropy_bytes(img),
                         chi2_p=np.nan, corr_H=co["H"], corr_V=co["V"], corr_D=co["D"], lossless=np.nan,
                         ct_overhead_bytes=0))
        for label, sch, sign in CONFIGS:
            if sign:
                continue
            pkg = hyb.encrypt(sch, data, shape)
            out = hyb.decrypt(sch, pkg)
            cimg = bytes_as_image(pkg["ct"], shape)
            c = adjacent_corr(cimg, rng)
            rows.append(dict(image=it["id"], scheme=label, ct_entropy=entropy_bytes(np.frombuffer(pkg["ct"], np.uint8)),
                             chi2_p=chi2_uniform(np.frombuffer(pkg["ct"], np.uint8))[1],
                             corr_H=c["H"], corr_V=c["V"], corr_D=c["D"], lossless=float(out == data),
                             ct_overhead_bytes=len(pkg["ct"]) - len(data)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Experiment 2: repeated timing of the hybrid schemes
# ----------------------------------------------------------------------------------------------
def exp_timing(items, hyb, args):
    rows = []
    for it in items[: args.timing_images]:
        data, shape = it["img"].tobytes(), it["img"].shape
        for label, sch, sign in CONFIGS:
            for _ in range(2):                                    # warm-up
                hyb.decrypt(sch, hyb.encrypt(sch, data, shape, sign), sign)
            gc.collect()
            gc.disable()
            for r in range(args.repeats):
                t0 = time.perf_counter()
                pkg = hyb.encrypt(sch, data, shape, sign)
                t1 = time.perf_counter()
                out = hyb.decrypt(sch, pkg, sign)
                t2 = time.perf_counter()
                assert out == data
                rows.append(dict(image=it["id"], config=label, rep=r, bytes=len(data),
                                 enc_ms=(t1 - t0) * 1e3, dec_ms=(t2 - t1) * 1e3))
            gc.enable()
        log(f"  timing: {it['id']} done")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Experiment 3: pure-RSA baselines (pixel-wise textbook RSA and chunked RSA-OAEP)
# ----------------------------------------------------------------------------------------------
def exp_rsa(items, hyb, args):
    rng = rng_for(args.seed, "rsa")
    key = hyb.recv_priv
    n, e, d = int(key.n), int(key.e), int(key.d)
    chunk = key.size_in_bytes() - 2 * SHA256.digest_size - 2                 # 190 bytes for RSA-2048/SHA-256
    rows, detail = [], []
    for it in items[: args.rsa_images]:
        img, data = it["img"], it["img"].tobytes()
        flat = img.ravel()
        N = flat.size
        # (a) pixel-wise textbook RSA (as in Anane et al.): one RSA operation per pixel
        for r in range(args.rsa_repeats):
            idx = rng.choice(N, size=min(args.rsa_enc_sample, N), replace=False)
            vals = flat[idx].tolist()
            t0 = time.perf_counter()
            cs = [pow(v, e, n) for v in vals]
            t_enc = time.perf_counter() - t0
            k = min(args.rsa_dec_sample, len(cs))
            t0 = time.perf_counter()
            ms_ = [pow(c, d, n) for c in cs[:k]]
            t_dec = time.perf_counter() - t0
            assert ms_ == vals[:k]
            per_enc, per_dec = t_enc / len(vals), t_dec / k
            rows.append(dict(image=it["id"], method="pixelwise textbook RSA (extrapolated)", rep=r, pixels=N,
                             enc_s=per_enc * N, dec_s=per_dec * N, enc_sample=len(vals), dec_sample=k,
                             enc_us_per_pixel=per_enc * 1e6, dec_us_per_pixel=per_dec * 1e6))
        if args.rsa_pixel_full_enc:
            vals = flat.tolist()
            t0 = time.perf_counter()
            _ = [pow(v, e, n) for v in vals]
            t_full = time.perf_counter() - t0
            rows.append(dict(image=it["id"], method="pixelwise textbook RSA (DIRECT, encryption only)", rep=0,
                             pixels=N, enc_s=t_full, dec_s=np.nan, enc_sample=N, dec_sample=0,
                             enc_us_per_pixel=t_full / N * 1e6, dec_us_per_pixel=np.nan))
        # (b) chunked RSA-OAEP of the entire byte stream, measured directly
        chunks = [data[i:i + chunk] for i in range(0, len(data), chunk)]
        for r in range(args.rsa_oaep_repeats):
            t0 = time.perf_counter()
            cts = [hyb.enc_c.encrypt(c) for c in chunks]
            t1 = time.perf_counter()
            pts = [hyb.dec_c.decrypt(c) for c in cts]
            t2 = time.perf_counter()
            assert b"".join(pts) == data
            rows.append(dict(image=it["id"], method="chunked RSA-OAEP (DIRECT)", rep=r, pixels=N,
                             enc_s=t1 - t0, dec_s=t2 - t1, enc_sample=len(chunks), dec_sample=len(chunks),
                             enc_us_per_pixel=(t1 - t0) / N * 1e6, dec_us_per_pixel=(t2 - t1) / N * 1e6))
        detail.append(f"{it['id']}: N={N} pixels; chunk={chunk} B -> {len(chunks)} OAEP blocks")
        log("  rsa baselines: " + detail[-1])
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Experiment 4: diffusion (NPCR / UACI) with fixed key+IV and with fresh keys
# ----------------------------------------------------------------------------------------------
def exp_diffusion(items, args):
    rng = rng_for(args.seed, "diffusion")
    rows = []
    for it in items[: args.sec_images]:
        data = it["img"].tobytes()
        for sch in [SCHEME_CBCH, SCHEME_GCM128, SCHEME_GCM256, SCHEME_CHACHA]:
            fixed_km = hashlib.sha512(f"{args.seed}-{sch.label}-km".encode()).digest()[: sch.km_len].ljust(sch.km_len, b"\0")
            fixed_iv = hashlib.sha256(f"{args.seed}-{sch.label}-iv".encode()).digest()[: sch.iv_len]
            c0, _ = sch.encrypt_sym(fixed_km, fixed_iv, data, b"")
            positions = [("first byte", 0)] + [("random byte", int(rng.integers(1, len(data)))) for _ in range(args.npcr_trials)]
            for pos_name, pos in positions:
                mod = bytearray(data)
                mod[pos] ^= 1 << int(rng.integers(0, 8))
                c1, _ = sch.encrypt_sym(fixed_km, fixed_iv, bytes(mod), b"")
                np_, ua_ = npcr_uaci(c0, c1)
                rows.append(dict(image=it["id"], scheme=sch.label, mode="fixed key+IV", flip=pos_name,
                                 pos_frac=pos / len(data), npcr=np_, uaci=ua_))
                # fresh independent keys/IV for both encryptions (ciphertext independence)
                km0, iv0, km1, iv1 = (get_random_bytes(sch.km_len), get_random_bytes(sch.iv_len),
                                      get_random_bytes(sch.km_len), get_random_bytes(sch.iv_len))
                a, _ = sch.encrypt_sym(km0, iv0, data, b"")
                b, _ = sch.encrypt_sym(km1, iv1, bytes(mod), b"")
                np_, ua_ = npcr_uaci(a, b)
                rows.append(dict(image=it["id"], scheme=sch.label, mode="fresh random key+IV", flip=pos_name,
                                 pos_frac=pos / len(data), npcr=np_, uaci=ua_))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Experiment 5: key sensitivity (raw decryption, NO unpadding fallback)
# ----------------------------------------------------------------------------------------------
def exp_key_sensitivity(items, hyb, args):
    rng = rng_for(args.seed, "keysens")
    rows = []
    for it in items[: args.sec_images]:
        img, data, shape = it["img"], it["img"].tobytes(), it["img"].shape
        for t in range(args.npcr_trials):
            key, iv = get_random_bytes(16), get_random_bytes(16)
            ct = AES.new(key, AES.MODE_CBC, iv).encrypt(pad(data, 16))
            kb = bytearray(key)
            bit = int(rng.integers(0, 128))
            kb[bit // 8] ^= 1 << (bit % 8)
            wrong = bytes_as_image(AES.new(bytes(kb), AES.MODE_CBC, iv).decrypt(ct), shape)
            rows.append(dict(image=it["id"], trial=t, test="1-bit wrong AES key, raw CBC decrypt (no unpad)",
                             pixels_different_pct=100 * float(np.mean(wrong != img)), mse=mse(img, wrong),
                             psnr=psnr(img, wrong), ssim=ssim(img, wrong), entropy=entropy_bytes(wrong),
                             rejected=np.nan))
        # authenticated schemes: flip one random bit of the whole key material -> must be rejected
        for sch in CORE_SCHEMES:
            rej = 0
            for _ in range(args.npcr_trials):
                km, iv = get_random_bytes(sch.km_len), get_random_bytes(sch.iv_len)
                ct, tag = sch.encrypt_sym(km, iv, data, Hybrid.aad(shape))
                kb = bytearray(km)
                bit = int(rng.integers(0, 8 * len(kb)))
                kb[bit // 8] ^= 1 << (bit % 8)
                try:
                    sch.decrypt_sym(bytes(kb), iv, ct, tag, Hybrid.aad(shape))
                except ValueError:
                    rej += 1
            rows.append(dict(image=it["id"], trial=-1, test=f"1-bit wrong key material -> {sch.label}",
                             pixels_different_pct=np.nan, mse=np.nan, psnr=np.nan, ssim=np.nan, entropy=np.nan,
                             rejected=rej / args.npcr_trials))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Experiment 6: corruption models + recovery (TELEA / Navier-Stokes / neighbour mean)
# ----------------------------------------------------------------------------------------------
def mask_random(shape, rate, rng):
    return rng.random(shape) < rate


def mask_blocks(shape, rate, rng, b=8):
    h, w = shape
    m = np.zeros(shape, bool)
    while m.mean() < rate:
        y, x = int(rng.integers(0, h - b + 1)), int(rng.integers(0, w - b + 1))
        m[y:y + b, x:x + b] = True
    return m


def mask_bursts(shape, rate, rng, length=64):
    h, w = shape
    m = np.zeros(shape, bool)
    L = min(length, w)
    while m.mean() < rate:
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w - L + 1))
        m[y, x:x + L] = True
    return m


def cipher_block_loss(img, rate, rng):
    """Lose `rate` of the 16-byte CBC ciphertext blocks (zeroed), decrypt without authentication.
    A lost block garbles itself AND the next plaintext block (CBC error propagation)."""
    h, w = img.shape
    data = img.tobytes()
    key, iv = get_random_bytes(16), get_random_bytes(16)
    ct = bytearray(AES.new(key, AES.MODE_CBC, iv).encrypt(pad(data, 16)))
    nb = len(ct) // 16
    lost = rng.choice(nb - 1, size=max(1, int(round(rate * nb))), replace=False)
    for i in lost:
        ct[16 * i:16 * i + 16] = b"\x00" * 16
    pt = AES.new(key, AES.MODE_CBC, iv).decrypt(bytes(ct))
    corrupted = np.frombuffer(pt[: h * w], np.uint8).reshape(h, w).copy()
    mask = np.zeros(h * w, bool)
    for i in lost:
        for blk in (i, i + 1):
            s, e = 16 * blk, min(16 * (blk + 1), h * w)
            if s < h * w:
                mask[s:e] = True
    return corrupted, mask.reshape(h, w)


def neighbour_mean_fill(img, mask):
    out = img.copy()
    todo = mask.copy()
    valid_all = (~mask).astype(np.float32)
    x = img.astype(np.float32) * valid_all
    for k in (5, 11, 21, 41, 81, 161):
        if not todo.any():
            break
        num, den = cv2.blur(x, (k, k)), cv2.blur(valid_all, (k, k))
        ok = todo & (den > 1e-6)
        out[ok] = np.clip(np.round(num[ok] / den[ok]), 0, 255).astype(np.uint8)
        todo &= ~ok
    return out


def exp_recovery(items, args, outdir):
    rng = rng_for(args.seed, "recovery")
    rows = []
    scenarios = [("random pixel loss", r) for r in (0.01, 0.05, 0.10, 0.20)] + \
                [("8x8 block loss", 0.05), ("burst loss (64 px runs)", 0.05), ("ciphertext CBC block loss", 0.05)]
    for ii, it in enumerate(items[: args.rec_images]):
        img = it["img"]
        for model, rate in scenarios:
            if model == "random pixel loss":
                mask = mask_random(img.shape, rate, rng)
                cor = img.copy(); cor[mask] = 0
            elif model == "8x8 block loss":
                mask = mask_blocks(img.shape, rate, rng)
                cor = img.copy(); cor[mask] = 0
            elif model.startswith("burst"):
                mask = mask_bursts(img.shape, rate, rng)
                cor = img.copy(); cor[mask] = 0
            else:
                cor, mask = cipher_block_loss(img, rate, rng)
            m8 = mask.astype(np.uint8) * 255
            outs = {"no recovery (as received)": cor,
                    "TELEA (r=3)": cv2.inpaint(cor, m8, 3, cv2.INPAINT_TELEA),
                    "Navier-Stokes (r=3)": cv2.inpaint(cor, m8, 3, cv2.INPAINT_NS),
                    "neighbour-mean baseline": neighbour_mean_fill(cor, mask)}
            for meth, rec in outs.items():
                diff = (rec.astype(np.float64) - img.astype(np.float64)) ** 2
                rows.append(dict(image=it["id"], model=model, nominal_rate=rate, coverage=float(mask.mean()),
                                 method=meth, mse=float(diff.mean()), psnr=psnr(img, rec), ssim=ssim(img, rec),
                                 mse_masked_region=float(diff[mask].mean()) if mask.any() else np.nan))
            if ii == 0 and args.save_figs and model in ("random pixel loss", "ciphertext CBC block loss") and rate == 0.05:
                tag = "ciphertextloss" if model.startswith("cipher") else "randompixel"
                slug = {"no recovery (as received)": "corrupted", "TELEA (r=3)": "telea",
                        "Navier-Stokes (r=3)": "ns", "neighbour-mean baseline": "neighbourmean"}
                for meth, rec in outs.items():
                    cv2.imwrite(os.path.join(outdir, "figs", f"{tag}_{slug[meth]}.png"), rec)
                cv2.imwrite(os.path.join(outdir, "figs", "original.png"), img)
        log(f"  recovery: {it['id']} done")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Experiment 7: tamper / forgery tests (why a plain SHA-256 is not authentication)
# ----------------------------------------------------------------------------------------------
def exp_tamper(items, hyb, args):
    rng = rng_for(args.seed, "tamper")
    img = items[0]["img"]
    data, shape = img.tobytes(), img.shape
    forged_img = np.clip(img.astype(np.int16) + 40, 0, 255).astype(np.uint8)     # attacker's forged image
    attacker_key = RSA.generate(2048)                                            # attacker's own signing key
    cfgs = [(l, s, sg) for (l, s, sg) in CONFIGS if s in (SCHEME_LEGACY, SCHEME_CBCH, SCHEME_GCM128)]
    rows = []

    def outcome(sch, pkg, sign):
        try:
            out = hyb.decrypt(sch, pkg, sign)
        except ValueError:
            return "rejected"
        if out == data and tuple(pkg["shape"]) == tuple(shape):
            return "accepted (unchanged)"
        if out == forged_img.tobytes():
            return "ACCEPTED FORGERY"
        return "ACCEPTED MODIFIED DATA"

    for label, sch, sign in cfgs:
        for attack in ("flip 1 ciphertext bit", "flip 1 IV bit", "tamper image-size header",
                       "full forgery with receiver PUBLIC key only"):
            res = []
            trials = 1 if attack.startswith(("full", "tamper")) else args.tamper_trials
            for _ in range(trials):
                pkg = hyb.encrypt(sch, data, shape, sign)
                if attack == "flip 1 ciphertext bit":
                    ct = bytearray(pkg["ct"]); b = int(rng.integers(0, len(ct) * 8)); ct[b // 8] ^= 1 << (b % 8)
                    pkg["ct"] = bytes(ct)
                elif attack == "flip 1 IV bit":
                    iv = bytearray(pkg["iv"]); b = int(rng.integers(0, len(iv) * 8)); iv[b // 8] ^= 1 << (b % 8)
                    pkg["iv"] = bytes(iv)
                elif attack == "tamper image-size header":
                    pkg["shape"] = (max(1, shape[0] // 2), shape[1] * 2)
                else:
                    pkg = hyb.encrypt(sch, forged_img.tobytes(), shape, False)       # attacker builds a valid package
                    if sign:
                        pkg["sig"] = hyb.sign(pkg, key=attacker_key)               # cannot use the sender's key
                res.append(outcome(sch, pkg, sign))
            for o in sorted(set(res)):
                rows.append(dict(config=label, attack=attack, outcome=o, fraction=res.count(o) / len(res), trials=len(res)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Summary tables (markdown) ready to paste into the paper / response letter
# ----------------------------------------------------------------------------------------------
def summarize(out, args, dfs, items):
    L = ["# Results summary (auto-generated)", "",
         f"Images analysed: **{len(items)}** (seed={args.seed}, resize={args.resize}). "
         f"Values are mean ± SD across images unless stated otherwise.", ""]
    if args.synthetic:
        L += ["> **WARNING: synthetic images – smoke test only. Do not report these numbers.**", ""]

    if "basic" in dfs:
        b = dfs["basic"]
        t = []
        for sch, g in b.groupby("scheme", sort=False):
            t.append({"Scheme": sch, "Entropy (bits/byte)": pm(g.ct_entropy, 4),
                      "Chi-sq uniformity p>0.05": "n/a" if g.chi2_p.isna().all() else f"{(g.chi2_p > 0.05).mean() * 100:.0f}% of images",
                      "|corr| H": pm(g.corr_H.abs(), 4), "|corr| V": pm(g.corr_V.abs(), 4), "|corr| D": pm(g.corr_D.abs(), 4),
                      "Lossless round trip": "n/a" if g.lossless.isna().all() else f"{g.lossless.mean() * 100:.0f}%",
                      "Ciphertext overhead (B)": pm(g.ct_overhead_bytes, 1)})
        L += ["## Entropy, uniformity, adjacent-pixel correlation, lossless round trip (Tables 3, 5, 9, 10)",
              md_table(pd.DataFrame(t)), ""]

    if "timing" in dfs:
        tm = dfs["timing"]
        t = []
        for cfg, g in tm.groupby("config", sort=False):
            e, d = g.enc_ms.values, g.dec_ms.values
            elo, ehi = ci95(e); dlo, dhi = ci95(d)
            t.append({"Configuration": cfg, "n": len(g),
                      "Enc ms (mean ± SD)": pm(e, 3), "Enc 95% CI": f"[{elo:.3f}, {ehi:.3f}]", "Enc median": f"{np.median(e):.3f}",
                      "Dec ms (mean ± SD)": pm(d, 3), "Dec 95% CI": f"[{dlo:.3f}, {dhi:.3f}]", "Dec median": f"{np.median(d):.3f}"})
        L += [f"## Hybrid timing, end-to-end incl. RSA-2048-OAEP key wrap (Tables 4, 9) – {args.repeats} repeats/image",
              md_table(pd.DataFrame(t)),
              "(Pooled over repeats and images; use --resize so all images have the same size.)", ""]

    if "rsa" in dfs:
        r = dfs["rsa"]
        t = []
        for (img_id, meth), g in r.groupby(["image", "method"], sort=False):
            t.append({"Image": img_id, "Baseline": meth, "Pixels": int(g.pixels.iloc[0]),
                      "Enc s (mean ± SD)": pm(g.enc_s, 3), "Dec s (mean ± SD)": pm(g.dec_s, 3) if g.dec_s.notna().any() else "n/a",
                      "Enc µs/pixel": f"{g.enc_us_per_pixel.mean():.2f}",
                      "Dec µs/pixel": f"{g.dec_us_per_pixel.mean():.1f}" if g.dec_us_per_pixel.notna().any() else "n/a"})
        L += ["## Pure-RSA baselines (Table 4)", md_table(pd.DataFrame(t)), ""]
        if "timing" in dfs:
            tm = dfs["timing"]
            sp = []
            for img_id in r.image.unique():
                for hcfg in (SCHEME_CBCH.label, SCHEME_GCM128.label):
                    g = tm[(tm.image == img_id) & (tm.config == hcfg)]
                    if g.empty:
                        continue
                    he, hd = g.enc_ms.mean() / 1e3, g.dec_ms.mean() / 1e3
                    for meth, gg in r.groupby("method"):
                        gg = gg[gg.image == img_id]
                        if gg.empty:
                            continue
                        be, bd = gg.enc_s.mean(), gg.dec_s.mean()
                        sp.append({"Image": img_id, "Hybrid": hcfg, "RSA baseline": meth,
                                   "Enc speed-up": f"x{be / he:,.0f}", "Dec speed-up": "n/a" if np.isnan(bd) else f"x{bd / hd:,.0f}"})
            if sp:
                L += ["## Speed-up of the hybrid scheme (ratio of mean times; order-of-magnitude figures)",
                      md_table(pd.DataFrame(sp)), ""]

    if "diffusion" in dfs:
        d = dfs["diffusion"]
        t = []
        for (sch, mode, flip), g in d.groupby(["scheme", "mode", "flip"], sort=False):
            t.append({"Scheme": sch, "Key/IV": mode, "1-bit plaintext flip at": flip,
                      "NPCR % (mean ± SD)": pm(g.npcr, 4), "UACI % (mean ± SD)": pm(g.uaci, 4), "n": len(g)})
        L += [f"## NPCR / UACI (ideal: NPCR {NPCR_IDEAL}, UACI {UACI_IDEAL}) (Table 11)",
              md_table(pd.DataFrame(t)),
              "Interpretation: with a FIXED key+IV, stream-like modes (GCM, ChaCha20) change only the flipped byte "
              "(NPCR≈0 is expected and not a weakness), while CBC diffuses forward from the flipped block. "
              "With fresh random keys NPCR/UACI only reflects ciphertext randomness, not diffusion.", ""]

    if "keysens" in dfs:
        k = dfs["keysens"]
        a = k[k.trial >= 0]
        L += ["## Key sensitivity (Section 6.9d)",
              f"1-bit wrong AES-128 key, raw CBC decrypt without unpadding, {len(a)} trials: "
              f"pixels differing {pm(a.pixels_different_pct, 2)} %, PSNR {pm(a.psnr, 2)} dB, SSIM {pm(a.ssim, 4)}, "
              f"entropy {pm(a.entropy, 4)} bits/byte.", ""]
        rj = k[k.trial < 0].copy()
        rj["scheme"] = rj.test.str.replace("1-bit wrong key material -> ", "", regex=False)
        t = [{"Scheme": s, "Wrong-key decryptions rejected": f"{g.rejected.mean() * 100:.1f}%"} for s, g in rj.groupby("scheme", sort=False)]
        L += [md_table(pd.DataFrame(t)), ""]

    if "recovery" in dfs:
        rc = dfs["recovery"]
        t = []
        for (model, rate, meth), g in rc.groupby(["model", "nominal_rate", "method"], sort=False):
            t.append({"Corruption": model, "Nominal rate": f"{rate * 100:.0f}%", "Actual coverage": f"{g.coverage.mean() * 100:.1f}%",
                      "Method": meth, "PSNR dB": pm(g.psnr, 2), "SSIM": pm(g.ssim, 4), "MSE": pm(g.mse, 2)})
        L += ["## Corruption models and recovery (Tables 6-8)", md_table(pd.DataFrame(t)),
              "Note: PSNR/SSIM are image-similarity metrics; they do not establish diagnostic acceptability.", ""]

    if "tamper" in dfs:
        L += ["## Tamper / forgery tests (SHA-256 vs MAC/AEAD vs signature)", md_table(dfs["tamper"].assign(
            fraction=lambda x: (x.fraction * 100).round(1).astype(str) + "%")), ""]

    with open(os.path.join(out, "paper_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ----------------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", help="folder with the X-ray images (searched recursively)")
    ap.add_argument("--synthetic", type=int, default=0, help="use N synthetic images (smoke test only)")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--max_images", type=int, default=100)
    ap.add_argument("--resize", type=lambda s: tuple(int(v) for v in s.lower().split("x")), default=None,
                    help="HxW, e.g. 415x720; recommended so that all images have the same size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=30, help="timing repeats per image and scheme")
    ap.add_argument("--timing_images", type=int, default=5)
    ap.add_argument("--sec_images", type=int, default=20, help="images used for NPCR/UACI and key sensitivity")
    ap.add_argument("--npcr_trials", type=int, default=10)
    ap.add_argument("--rec_images", type=int, default=20, help="images used for corruption/recovery")
    ap.add_argument("--tamper_trials", type=int, default=20)
    ap.add_argument("--rsa_images", type=int, default=2)
    ap.add_argument("--rsa_enc_sample", type=int, default=4096)
    ap.add_argument("--rsa_dec_sample", type=int, default=128)
    ap.add_argument("--rsa_repeats", type=int, default=5)
    ap.add_argument("--rsa_oaep_repeats", type=int, default=3)
    ap.add_argument("--rsa_pixel_full_enc", action="store_true",
                    help="also measure DIRECT full-image pixel-wise RSA encryption (~tens of seconds)")
    ap.add_argument("--skip", default="", help="comma list: basic,timing,rsa,diffusion,keysens,recovery,tamper")
    ap.add_argument("--no_figs", dest="save_figs", action="store_false")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    a = ap.parse_args()
    if a.quick:
        a.repeats, a.timing_images, a.sec_images, a.npcr_trials, a.rec_images = 5, 2, 2, 3, 2
        a.tamper_trials, a.rsa_images, a.rsa_enc_sample, a.rsa_dec_sample, a.rsa_repeats, a.rsa_oaep_repeats = 5, 1, 512, 16, 2, 1
    if not a.synthetic and not a.data_dir:
        ap.error("provide --data_dir (or --synthetic N for a smoke test)")
    return a


def main():
    args = parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    os.makedirs(os.path.join(args.out_dir, "figs"), exist_ok=True)
    items = load_images(args, rng_for(args.seed, "select"))
    log(f"Loaded {len(items)} image(s); shapes: {sorted({i['img'].shape for i in items})[:5]}")
    with open(os.path.join(args.out_dir, "environment.json"), "w") as f:
        json.dump(environment(args, items), f, indent=2)

    t0 = time.perf_counter()
    recv, sender = RSA.generate(2048), RSA.generate(2048)
    log(f"Generated 2 RSA-2048 key pairs in {time.perf_counter() - t0:.2f}s")
    hyb = Hybrid(recv, sender)

    jobs = [("basic", lambda: exp_basic(items, hyb, args)),
            ("timing", lambda: exp_timing(items, hyb, args)),
            ("rsa", lambda: exp_rsa(items, hyb, args)),
            ("diffusion", lambda: exp_diffusion(items, args)),
            ("keysens", lambda: exp_key_sensitivity(items, hyb, args)),
            ("recovery", lambda: exp_recovery(items, args, args.out_dir)),
            ("tamper", lambda: exp_tamper(items, hyb, args))]
    dfs = {}
    for name, fn in jobs:
        if name in skip:
            continue
        log(f"[{name}] running ...")
        t = time.perf_counter()
        dfs[name] = fn()
        dfs[name].to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
        log(f"[{name}] done in {time.perf_counter() - t:.1f}s -> {name}.csv")
    summarize(args.out_dir, args, dfs, items)
    log(f"\nAll done. Open {os.path.join(args.out_dir, 'paper_tables.md')}")


if __name__ == "__main__":
    main()
