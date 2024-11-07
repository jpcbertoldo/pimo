#!/usr/bin/env python

# %%
# Setup (pre args)

from __future__ import annotations

import argparse
import sys
import warnings
from functools import partial
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import torch
from PIL import Image

from aupimo.oracles import IOUCurvesResult, MaxIOUPerImageResult
from aupimo.oracles_numpy import (
    open_image,
    upscale_image_asmap_mask,
    valid_anomaly_score_maps,
)

# is it running as a notebook or as a script?
if (arg0 := Path(sys.argv[0]).stem) == "ipykernel_launcher":
    print("running as a notebook")
    from IPython import get_ipython
    from IPython.core.interactiveshell import InteractiveShell

    IS_NOTEBOOK = True

    # autoreload modified modules
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
    # make a cell print all the outputs instead of just the last one
    InteractiveShell.ast_node_interactivity = "all"
    # show all warnings
    warnings.filterwarnings("always", category=Warning)

else:
    IS_NOTEBOOK = False


from aupimo._validate_tensor import safe_tensor_to_numpy
from aupimo.oracles_numpy import (
    calculate_levelset_mean_dist_to_superpixel_boundaries_curve,
)

# %%
# Args

# collection [of datasets] = {mvtec, visa}
ACCEPTED_COLLECTIONS = {
    (MVTEC_DIR_NAME := "MVTec"),
    (VISA_DIR_NAME := "VisA"),
}

parser = argparse.ArgumentParser()
_ = parser.add_argument("--asmaps", type=Path, required=True)
_ = parser.add_argument("--mvtec-root", type=Path)
_ = parser.add_argument("--visa-root", type=Path)
_ = parser.add_argument("--not-debug", dest="debug", action="store_false")
_ = parser.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")
_ = parser.add_argument("--savedir", type=Path, default=None)

if IS_NOTEBOOK:
    print("argument string")
    print(
        argstrs := [
            string
            for arg in [
                # "--asmaps ../../../data/experiments/benchmark/efficientad_wr101_s_ext/mvtec/capsule/asmaps.pt",
                "--asmaps ../../../data/experiments/benchmark/efficientad_wr101_s_ext/mvtec/screw/asmaps.pt",
                # "--asmaps ../../../data/experiments/benchmark/efficientad_wr101_s_ext/mvtec/transistor/asmaps.pt",
                # "--asmaps ../../../data/experiments/benchmark/rd++_wr50_ext/mvtec/bottle/asmaps.pt",
                "--mvtec-root ../../../data/datasets/MVTec",
                "--visa-root ../../../data/datasets/VisA",
                "--not-debug",
                "--savedir /home/jcasagrandebertoldo/repos/2024-these/ch2b/img/",
            ]
            for string in arg.split(" ")
        ],
    )
    args = parser.parse_args(argstrs)

else:
    args = parser.parse_args()

print(f"{args=}")

rundir = args.asmaps.parent
assert rundir.exists(), f"{rundir=}"

if args.savedir is not None:
    assert args.savedir.exists(), f"{args.savedir=}"

# %%
# Load `asmaps.pt`

print("loading asmaps.pt")

assert args.asmaps.exists(), str(args.asmaps)

asmaps_dict = torch.load(args.asmaps)
assert isinstance(asmaps_dict, dict), f"{type(asmaps_dict)=}"

asmaps = asmaps_dict["asmaps"]
assert isinstance(asmaps, torch.Tensor), f"{type(asmaps)=}"
assert asmaps.ndim == 3, f"{asmaps.shape=}"
print(f"{asmaps.shape=}")

images_relpaths = asmaps_dict["paths"]
assert isinstance(images_relpaths, list), f"{type(images_relpaths)=}"

assert len(asmaps) == len(images_relpaths), f"{len(asmaps)=}, {len(images_relpaths)=}"

# collection [of datasets] = {mvtec, visa}
collection = {p.split("/")[0] for p in images_relpaths}
assert collection.issubset(ACCEPTED_COLLECTIONS), f"{collection=}"

collection = collection.pop()

if collection == MVTEC_DIR_NAME:
    assert args.mvtec_root is not None, "please provide the argument `--mvtec-root`"
    collection_root = args.mvtec_root

if collection == VISA_DIR_NAME:
    assert args.visa_root is not None, "please provide the argument `--visa-root`"
    collection_root = args.visa_root

assert collection_root.exists(), f"{collection=} {collection_root=!s}"

dataset = {"/".join(p.split("/")[:2]) for p in images_relpaths}
assert len(dataset) == 1, f"{dataset=}"

dataset = dataset.pop()
print(f"{dataset=}")

print("sorting images and their asmaps")
images_argsort = np.argsort(images_relpaths)
images_relpaths = np.array(images_relpaths)[images_argsort].tolist()
asmaps = asmaps[images_argsort]

print("getting masks paths from images paths")


def _image_path_2_mask_path(image_path: str) -> str | None:
    if "good" in image_path:
        # there is no mask for the normal images
        return None

    path = Path(image_path.replace("test", "ground_truth"))

    if (collection := path.parts[0]) == VISA_DIR_NAME:
        path = path.with_suffix(".png")

    elif collection == MVTEC_DIR_NAME:
        path = path.with_stem(path.stem + "_mask").with_suffix(".png")

    else:
        msg = f"Unknown collection: {collection=}"
        raise NotImplementedError(msg)

    return str(path)


masks_relpaths = [_image_path_2_mask_path(p) for p in images_relpaths]

print(f"converting relative paths to absolute paths\n{collection_root=!s}")


def _convert_path(relative_path: str, collection_root: Path) -> str | None:
    if relative_path is None:
        return None
    relative_path = Path(*Path(relative_path).parts[1:])
    return str(collection_root / relative_path)


_convert_path = partial(_convert_path, collection_root=collection_root)

images_abspaths = [_convert_path(p) for p in images_relpaths]
masks_abspaths = [_convert_path(p) for p in masks_relpaths]

for path in images_abspaths + masks_abspaths:
    assert path is None or Path(path).exists(), path

# %%
# Load masks

print("loading masks")
masks_pils = [Image.open(p).convert("L") if p is not None else None for p in masks_abspaths]

masks_resolution = {p.size for p in masks_pils if p is not None}
assert len(masks_resolution) == 1, f"assumed single-resolution dataset but found {masks_resolution=}"
masks_resolution = masks_resolution.pop()
masks_resolution = (masks_resolution[1], masks_resolution[0])  # [W, H] --> [H, W]
print(f"{masks_resolution=} (HEIGHT, WIDTH)")

masks = torch.stack(
    [
        torch.tensor(np.asarray(pil), dtype=torch.bool)
        if pil is not None
        else torch.zeros(masks_resolution, dtype=torch.bool)
        for pil in masks_pils
    ],
    dim=0,
)
print(f"{masks.shape=}")

# %%
# DEBUG: only keep 2 images per class if in debug mode
if args.debug:
    print("debug mode --> only using 2 images")
    imgclass = (masks == 1).any(dim=-1).any(dim=-1).to(torch.bool)
    NUM_IMG_PER_CLASS = 5
    some_norm = torch.where(imgclass == 0)[0][:NUM_IMG_PER_CLASS]
    some_anom = torch.where(imgclass == 1)[0][:NUM_IMG_PER_CLASS]
    some_imgs = torch.cat([some_norm, some_anom])
    asmaps = asmaps[some_imgs]
    masks = masks[some_imgs]
    images_relpaths = [images_relpaths[i] for i in some_imgs]
    images_abspaths = [images_abspaths[i] for i in some_imgs]
    masks_abspaths = [masks_abspaths[i] for i in some_imgs]

# %%
# Resize asmaps to match the resolution of the masks

asmaps_resolution = asmaps.shape[-2:]
print(f"{asmaps_resolution=} (HEIGHT, WIDTH)")

if asmaps_resolution == masks_resolution:
    print("asmaps and masks have the same resolution")
else:
    print("resizing asmaps to match the resolution of the masks")
    asmaps = torch.nn.functional.interpolate(
        asmaps.unsqueeze(1),
        size=masks_resolution,
        mode="bilinear",
    ).squeeze(1)
    print(f"{asmaps.shape=}")

# %%
# Move data to device

if args.device == "cpu":
    print("using CPU")

elif args.device in ("cuda", "gpu"):
    print("moving data to GPU")
    masks = masks.cuda()
    asmaps = asmaps.cuda()

else:
    msg = f"Unknown device: {args.device=}"
    raise NotImplementedError(msg)

# %%

iou_oracle_threshs_dir = rundir / "iou_oracle_threshs"
superpixel_bound_dist_heuristic_dir = rundir / "superpixel_bound_dist_heuristic"
superpixel_oracle_selection_dir = Path(
    "/".join(rundir.parts[:-3] + ("patchcore_wr50",) + rundir.parts[-2:] + ("superpixel_oracle_selection",)),
)


# %%
ioucurves = IOUCurvesResult.load(iou_oracle_threshs_dir / "ioucurves_local_threshs.pt")
max_iou_per_image_result = MaxIOUPerImageResult.load(iou_oracle_threshs_dir / "max_iou_per_img_min_thresh.json")

payload_loaded = torch.load(superpixel_bound_dist_heuristic_dir / "superpixel_bound_dist_heuristic.pt")

# %%

# -------------------------------------
# worst to best argsort

# cable
# [142, 146, 149, 144, 141, 148, 145,  53,  54,  44,  52,  46,  58, 143,42,  50,  37, 147,  57,  43,  59,  36,  41,  39,  49,   5, 140,  48,10,  45,  40,   4, 132,   3,  51,  29,  38,   1, 137,  20,   9,   0,130,  47,  25,  33,  12,  27,  28,   8,  55,  56,  13, 136,   7,  34,11,   2,  26,  16, 133, 134,  24,  19,  18,  22, 135,   6,  23,  14,17,  32,  21, 131, 138, 123,  15,  30, 139,  35, 129, 126, 122, 121,31, 127, 120, 125, 118, 119, 124, 128]

# pill
# [147, 87, 12, 16, 71, 83, 21, 11, 79, 2, 17, 35, 7, 45, 25, 32, 29, 28, 166, 42, 30, 14, 76, 20, 64, 40, 65, 31, 9, 13, 165, 160, 6, 73, 93, 98, 67, 39, 37, 102, 149, 88, 91, 34, 55, 162, 10, 23, 15, 74, 158, 70, 143, 56, 27, 95, 148, 47, 1, 33, 82, 44, 97, 154, 61, 159, 41, 101, 144, 53, 26, 52, 5, 146, 49, 36, 163, 164, 69, 92, 77, 3, 38, 19, 75, 90, 72, 84, 78, 155, 85, 50, 68, 153, 0, 107, 57, 58, 66, 43, 48, 59, 63, 4, 81, 60, 105, 51, 89, 103, 152, 151, 100, 46, 80, 54, 104, 24, 94, 18, 86, 96, 157, 106, 161, 8, 145, 150, 99, 62, 156, 22, 137, 138, 136, 141, 142, 140, 139, 135, 134]

# screw
# [118, 43, 116, 127, 53, 123, 64, 114, 60, 84, 59, 124, 125, 72, 136, 52, 128, 133, 134, 131, 117, 94, 119, 129, 132, 121, 47, 49, 115, 58, 54, 76, 126, 122, 46, 152, 50, 48, 130, 159, 56, 120, 135, 95, 149, 150, 151, 110, 55, 44, 51, 65, 148, 62, 45, 68, 144, 112, 140, 87, 154, 41, 42, 158, 88, 71, 145, 111, 102, 113, 147, 80, 93, 83, 146, 74, 91, 57, 99, 156, 82, 142, 153, 100, 98, 139, 92, 109, 141, 67, 157, 108, 75, 70, 73, 81, 137, 79, 69, 155, 86, 78, 77, 143, 61, 66, 106, 103, 63, 104, 85, 138, 89, 96, 107, 97, 105, 101, 90]

# -------------------------------------
image_idx = 142

threshs = payload_loaded["threshs_per_image"][image_idx]
num_levelsets = threshs.shape[0]
min_thresh = payload_loaded["min_thresh"]
upscale_factor = payload_loaded["upscale_factor"]

local_minima_idxs = payload_loaded["local_minima_idxs_per_image"][image_idx][:5]
local_minima_threshs = threshs[local_minima_idxs]
local_minima_ious = ioucurves.per_image_ious[image_idx][
    np.argmin(np.abs(local_minima_threshs[None, ...] - ioucurves.threshs[image_idx][..., None]), axis=0)
]

watershed_superpixel_relsize = payload_loaded["superpixels_params"]["superpixel_relsize"]
watershed_compactness = payload_loaded["superpixels_params"]["compactness"]

img = open_image(_convert_path(payload_loaded["paths"][image_idx]))
mask = safe_tensor_to_numpy(masks[image_idx])
asmap = safe_tensor_to_numpy(asmaps[image_idx])
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# resize image and asmap to double the resolution
img, asmap, mask = upscale_image_asmap_mask(img, asmap, mask, upscale_factor=upscale_factor)
valid_asmap, valid_asmap_mask = valid_anomaly_score_maps(asmap[None, ...], min_thresh, return_mask=True)
valid_asmap = valid_asmap[0]
valid_asmap_mask = valid_asmap_mask[0]
from skimage.filters import sobel
# valid_asmap_sobel = sobel(image=asmap, mask=valid_asmap_mask)
valid_asmap_sobel = sobel(image=asmap)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
_ = ax.set_xticks([])
_ = ax.set_yticks([])

imshow = ax.imshow(img)
cs_gt = ax.contour(
    mask,
    levels=[0.5],
    colors="magenta",
    linewidths=3.5,
    linestyles="--",
)
_ = ax.contour(
    asmap,
    levels=[local_minima_threshs[2]],
    colors=["orange"],
    linewidths=3.5,
)
_ = ax.contour(
    asmap,
    levels=[local_minima_threshs[0]],
    colors=["yellow"],
    linewidths=3.5,
)
_ = ax.contour(
    asmap,
    levels=[local_minima_threshs[4]],
    colors=["red"],
    linewidths=3.5,
)
ax.annotate(
    f"IoU (oracle/yellow/orange/red): {max_iou_per_image_result.ious[image_idx].item():.0%} / {local_minima_ious[0]:.0%} / {local_minima_ious[1]:.0%} / {local_minima_ious[2]:.0%}",
    xy=(0, 0),
    xycoords="axes fraction",
    xytext=(10, 10),
    textcoords="offset points",
    ha="left",
    va="bottom",
    fontsize=19,
    bbox=dict(  # noqa: C408
        facecolor="white",
        alpha=1,
        edgecolor="black",
        boxstyle="round,pad=0.2",
    ),
)

# %%
fig.savefig(args.savedir / f"good_bad_{dataset.replace('/', '_')}_{image_idx}.jpg", bbox_inches="tight", pad_inches=0, dpi=100)

# %%
asort = max_iou_per_image_result.ious.argsort()
asort[torch.isin(asort, torch.where(~torch.isnan(max_iou_per_image_result.ious))[0])]

