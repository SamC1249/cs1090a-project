"""Download TNG100-1 z=0 group catalog and build the central-galaxy dataset.

The API does not support field selection on this catalog, so we stream the 448
chunk files one at a time, pull out only the columns we need, and delete each
chunk right away — nothing big is ever kept on disk. The final output is a
small CSV of central galaxies (~MBs), well under the project's 1GB data budget.

Usage:
    python src/download_data.py [--workers 6]
"""

import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "_chunk_cache"  # per-chunk extracted columns (tiny npz files)
OUT_PATH = DATA_DIR / "tng100-1_snap99_centrals.csv"

CHUNK_URL = "https://www.tng-project.org/api/TNG100-1/files/groupcat-99.{i}.hdf5"
N_CHUNKS = 448
HUBBLE = 0.6774  # TNG (Planck 2015) little-h

# columns extracted per chunk; 9-vector fields are sliced to the stellar entry (4)
GROUP_COLS = ["GroupFirstSub", "Group_M_Crit200", "Group_R_Crit200"]
BANDS = ["U", "B", "V", "K", "g", "r", "i", "z"]
MIN_LOG_MSTAR = 8.5  # ~350 star particles; below this properties are noisy


def load_api_key() -> str:
    env_path = REPO_ROOT / ".env"
    for line in env_path.read_text().splitlines():
        if line.startswith("TNG_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("TNG_API_KEY not found in .env")


def extract_chunk(i: int, api_key: str) -> Path:
    """Download chunk i, keep only needed columns, save tiny npz, delete chunk."""
    out = CACHE_DIR / f"chunk_{i:03d}.npz"
    if out.exists():
        return out
    for attempt in range(4):
        try:
            r = requests.get(
                CHUNK_URL.format(i=i), headers={"api-key": api_key}, timeout=300
            )
            r.raise_for_status()
            break
        except Exception:
            if attempt == 3:
                raise
    with tempfile.NamedTemporaryFile(suffix=".hdf5", dir=CACHE_DIR) as tmp:
        tmp.write(r.content)
        tmp.flush()
        cols = {}
        with h5py.File(tmp.name, "r") as f:
            g, s = f.get("Group"), f.get("Subhalo")
            for k in GROUP_COLS:
                cols[k] = g[k][:] if g is not None and k in g else np.array([])
            if s is not None and "SubhaloVmax" in s:
                cols["flag"] = s["SubhaloFlag"][:]
                cols["mstar"] = s["SubhaloMassInRadType"][:, 4]
                cols["sfr"] = s["SubhaloSFRinRad"][:]
                cols["gas_z"] = s["SubhaloGasMetallicity"][:]
                cols["star_z"] = s["SubhaloStarMetallicity"][:]
                cols["size"] = s["SubhaloHalfmassRadType"][:, 4]
                cols["phot"] = s["SubhaloStellarPhotometrics"][:]
                cols["vmax"] = s["SubhaloVmax"][:]
                cols["veldisp"] = s["SubhaloVelDisp"][:]
            else:
                cols["flag"] = np.array([])
    np.savez_compressed(out, **cols)
    return out


def download_all(api_key: str, workers: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(lambda i: extract_chunk(i, api_key), range(N_CHUNKS)):
            done += 1
            if done % 25 == 0 or done == N_CHUNKS:
                print(f"  {done}/{N_CHUNKS} chunks processed", flush=True)


def concat_field(name: str) -> np.ndarray:
    parts = []
    for i in range(N_CHUNKS):
        with np.load(CACHE_DIR / f"chunk_{i:03d}.npz") as z:
            if name not in z.files:
                continue
            a = z[name]
            if a.size:
                parts.append(a)
    return np.concatenate(parts)


def build_dataset() -> pd.DataFrame:
    first_sub = concat_field("GroupFirstSub").astype(np.int64)
    m200 = concat_field("Group_M_Crit200")
    r200 = concat_field("Group_R_Crit200")

    sub = {
        k: concat_field(k)
        for k in ["flag", "mstar", "sfr", "gas_z", "star_z", "size", "vmax", "veldisp"]
    }
    phot = concat_field("phot")

    has_sub = first_sub >= 0
    central_idx = first_sub[has_sub]
    m200, r200 = m200[has_sub], r200[has_sub]

    mass_to_msun = 1e10 / HUBBLE  # catalog masses are 1e10 Msun/h
    mstar = sub["mstar"][central_idx] * mass_to_msun
    mhalo = m200 * mass_to_msun

    ok = (sub["flag"][central_idx] == 1) & (mstar > 10**MIN_LOG_MSTAR) & (mhalo > 0)
    idx = central_idx[ok]

    sfr = sub["sfr"][idx]
    df = pd.DataFrame(
        {
            "subhalo_id": idx,
            "log_mhalo": np.log10(mhalo[ok]),
            "log_r200": np.log10(r200[ok] / HUBBLE),
            "log_mstar": np.log10(mstar[ok]),
            "log_size_star": np.log10(sub["size"][idx] / HUBBLE),  # ckpc/h -> kpc
            "sfr": sfr,
            "log_ssfr": np.log10(np.maximum(sfr, 1e-5) / mstar[ok]),
            "gas_metallicity": sub["gas_z"][idx],
            "star_metallicity": sub["star_z"][idx],
            "vmax": sub["vmax"][idx],
            "veldisp": sub["veldisp"][idx],
            **{f"mag_{b}": phot[idx, i] for i, b in enumerate(BANDS)},
        }
    )
    # standard observable colors
    df["color_gr"] = df["mag_g"] - df["mag_r"]
    df["color_ri"] = df["mag_r"] - df["mag_i"]
    df["color_UV"] = df["mag_U"] - df["mag_V"]
    df["color_VK"] = df["mag_V"] - df["mag_K"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    download_all(load_api_key(), args.workers)
    df = build_dataset()
    df.to_csv(OUT_PATH, index=False)
    print(f"saved {len(df):,} central galaxies -> {OUT_PATH} "
          f"({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
