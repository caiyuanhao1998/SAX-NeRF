"""Convert the released Backpack raw CT into the MAT input used by the generator."""
import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from scipy.io import savemat


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=Path,
                        default=root / "raw_data/backpack_512x512x373_uint16.raw",
                        help="path to the downloaded 512x512x373 uint16 raw CT")
    parser.add_argument("--output_folder", type=Path, default=root,
                        help="folder for CT_vis/backpack and raw_data/backpack/img.mat")
    args = parser.parse_args()
    raw = np.fromfile(args.raw_path, dtype=np.uint16)
    expected = 512 * 512 * 373
    if raw.size != expected:
        raise ValueError(f"Expected {expected} uint16 values in {args.raw_path}, got {raw.size}")
    volume = raw.reshape(373, 512, 512).transpose(1, 2, 0).astype(np.float32)
    peak = float(volume.max())
    if peak > 0:
        volume /= peak
    vis_dir = args.output_folder / "CT_vis/backpack"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for index, z in enumerate(np.linspace(0, volume.shape[-1] - 1, 51, dtype=int)):
        iio.imwrite(vis_dir / f"CT_raw_{index}.png", (volume[..., z] * 255).astype(np.uint8))
    output = args.output_folder / "raw_data/backpack/img.mat"
    output.parent.mkdir(parents=True, exist_ok=True)
    savemat(output, {"img": volume})
    print(f"Saved CT {volume.shape} to {output}")


if __name__ == "__main__":
    main()
