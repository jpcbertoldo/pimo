"""Oracle functions for evaluating anomaly detection algorithms.

TODO: write docstring of module.
TODO(jpcbertoldo): test this module.
"""

import copy
import logging
from functools import partial

import matplotlib as mpl
import numpy as np
import skimage as sk
from numpy import ndarray

from . import _validate
from .binclf_curve_numpy import BinclfAlgorithm, BinclfThreshsChoice, per_image_binclf_curve, per_image_iou

logger = logging.getLogger(__name__)


def per_image_iou_curves(
    anomaly_maps: ndarray,
    masks: ndarray,
    num_threshs: int,
    common_threshs: bool,
    binclf_algorithm: str = BinclfAlgorithm.NUMBA,
) -> tuple[ndarray, ndarray]:
    """TODO write docstring of `iou_curves`."""
    _validate.anomaly_maps(anomaly_maps)
    _validate.masks(masks)
    _validate.num_threshs(num_threshs)
    BinclfAlgorithm.validate(binclf_algorithm)

    threshs, binclfs = per_image_binclf_curve(
        anomaly_maps,
        masks,
        algorithm=binclf_algorithm,
        threshs_choice=(
            BinclfThreshsChoice.MINMAX_LINSPACE if common_threshs else BinclfThreshsChoice.MINMAX_LINSPACE_PER_IMAGE
        ),
        num_threshs=num_threshs,
    )

    ious = per_image_iou(binclfs)

    return threshs, ious


def get_superpixels_watershed(
    image: ndarray,
    superpixel_relsize: float = 1e-4,
    compactness: float = 0.001,
) -> ndarray:
    """TODO(jpcbertoldo): write docstring of `get_superpixels_watershed`."""
    num_markers = int(1 / superpixel_relsize)
    gradient = sk.filters.sobel(sk.color.rgb2gray(image))
    return sk.segmentation.watershed(
        gradient,
        markers=num_markers,
        # makes it harder for markers to flood faraway pixels
        # --> regions more regularly shaped
        compactness=compactness,
    )


def plot_superpixels_boundaries_vs_gt(
    ax: mpl.axes.Axes,
    image: ndarray,
    superpixels: ndarray,
    mask: ndarray,
) -> tuple[mpl.image.AxesImage, mpl.contour.QuadContourSet]:
    """TODO(jpcbertoldo): write docstring of `plot_superpixels_boundaries_vs_gt`."""
    imshow = ax.imshow(
        sk.segmentation.mark_boundaries(image, superpixels, color=mpl.colors.to_rgb("magenta")),
    )
    cs_gt = ax.contour(
        mask,
        levels=[0.5],
        colors="black",
        linewidths=2.5,
        linestyles="--",
    )
    return imshow, cs_gt


def _get_num_superpixels(superpixels: ndarray) -> int:
    return len(set(np.unique(superpixels)) - {0})


def plot_superpixels_colored_vs_gt(
    ax: mpl.axes.Axes,
    superpixels: ndarray,
    mask: ndarray,
) -> tuple[mpl.image.AxesImage, mpl.contour.QuadContourSet]:
    """TODO write docstring of `plot_superpixels_colored_vs_gt`."""
    ts = np.linspace(0, 1, 40)
    ts = np.concatenate(list(zip((ts_even := ts[0::2]), (ts_odd := ts[1::2][::-1]))))  # noqa: B905, F841
    colors = list(map(tuple, mpl.colormaps["jet"](ts)))

    imshow = ax.imshow(sk.color.label2rgb(superpixels, colors=colors))
    cs_gt = ax.contour(
        mask,
        levels=[0.5],
        colors="black",
        linewidths=2.5,
        linestyles="--",
    )
    return imshow, cs_gt


def superpixel_selection_map(
    superpixels: ndarray,
    selected_labels: set[int],
    available_labels: set[int],
) -> ndarray:
    """TODO(jpcbertoldo): write docstring of `suppix_selection_map`.

    Label meanings:
    - 0: ignored
    - 1: selected
    - 2: available
    """
    # 0 means "out of scope", meaning it is not in the initial `segments`
    viz = np.full_like(superpixels, np.nan, dtype=float)
    # 1 means "selected"
    viz[np.isin(superpixels, list(selected_labels))] = 1
    # 2 means "available"
    viz[np.isin(superpixels, list(available_labels))] = 2
    return viz


def find_best_superpixels(superpixels: ndarray, mask: ndarray) -> tuple[set[int], set[int]]:
    """TODO(jpcbertoldo): write docstring of `find_best_superpixels`."""

    def get_initial_state(superpixels: ndarray, mask: ndarray) -> tuple[set[int], set[int]]:
        # superpixels touching the ground truth mask
        on_gt_labels = set(np.unique(superpixels * mask.astype(int))) - {0}
        # supperpixels 100% inside the ground truth mask (so 0 False Positives)
        in_gt_labels = sorted(
            {label for label in on_gt_labels if (mask[superpixels == label]).all()},
        )
        return set(in_gt_labels), set(on_gt_labels) - set(in_gt_labels)

    # each one is a set of superpixels labels
    initial_selected, initial_available = get_initial_state(superpixels, mask)
    selected = copy.deepcopy(initial_selected)
    available = copy.deepcopy(initial_available)

    def get_binclfs_per_suppix(superpixels: ndarray, mask: ndarray, labels: set[int]) -> ndarray:
        """Make the computation of IoU faster (function below)."""
        num_superpixels = _get_num_superpixels(superpixels)
        # +1 in the shape is for the background mask
        # the -1 values are dummy values (not used)
        binclf_per_suppix = -np.ones((num_superpixels + 1, 2, 2), dtype=int)

        for label in labels:
            suppix_mask = superpixels == label
            tp = (mask & suppix_mask).sum()
            fp = (~mask & suppix_mask).sum()
            tn = (~mask & ~suppix_mask).sum()
            fn = (mask & ~suppix_mask).sum()
            binclf_per_suppix[label, :] = np.array([[tn, fp], [fn, tp]])

        return binclf_per_suppix

    def iou(selected: set[int], binclf_per_suppix: ndarray, mask: ndarray) -> float:
        binclfs = binclf_per_suppix[list(selected)]
        tp = binclfs[:, 1, 1].sum()
        fp = binclfs[:, 0, 1].sum()
        gt_size = mask.sum()
        return float(tp / (fp + gt_size))

    binclf_per_suppix = get_binclfs_per_suppix(
        superpixels,
        mask,
        # the binclf is not necessary for all superpixels, only the selected ones
        # and the available ones
        labels=initial_selected | initial_available,
    )

    iou_ = partial(
        iou,
        binclf_per_suppix=binclf_per_suppix,
        mask=mask,
    )

    history = [
        {
            "selected": copy.deepcopy(selected),
            "iou": iou_(selected),
        },
    ]

    # best first search
    for _ in range(len(initial_available) + 1):
        if len(available) == 0:
            return history, selected, available

        # find the segment that maximizes the iou of the current segmentation
        chosen = max(
            available,
            key=lambda candidate: iou_(selected | {candidate}),
        )

        selected.add(chosen)
        available.remove(chosen)
        history.append(
            {
                "chosen": int(chosen),
                "iou": iou_(selected),
            },
        )
        # when the iou decreases
        if history[-1]["iou"] < history[-2]["iou"]:
            # remove the last element and stop
            _ = history.pop()
            break

    return history, {int(idx) for idx in selected}, {int(idx) for idx in available}


def plot_superpixel_search_history(
    ax: mpl.axes.Axes,
    history: list[dict[str, set[int] | float]],
    asmap: ndarray,
    superpixels: ndarray,
    min_ascore: float | None = None,
) -> None:
    """TODO(jpcbertoldo): write docstring of `plot_superpixel_search_history`."""
    num_suppixs_hist = len(history[0]["selected"]) + np.arange(len(history))
    iou_hist = np.array([h["iou"] for h in history])

    # chosen history (intial step not in there)
    chosen_hist = np.array([h["chosen"] for h in history[1:]])
    chosen_candidate_mean_ascore_hist = np.array([asmap[superpixels == chosen].mean().item() for chosen in chosen_hist])

    initial_iou = history[0]["iou"]
    best_iou = history[-1]["iou"]

    _ = ax.plot(num_suppixs_hist, iou_hist, label="iou history", marker="o", markersize=3, color="black")
    _ = ax.axhline(best_iou, color="black", linestyle="--", label="best iou")
    _ = ax.axhline(initial_iou, color="black", linestyle="--", label="initial iou")

    _ = ax.set_xlabel("Number of superpixels selected")
    # integer format
    _ = ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    _ = ax.set_ylabel("IoU")
    _ = ax.set_ylim(0, 1)
    _ = ax.set_yticks(np.linspace(0, 1, 6))
    _ = ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    _ = ax.grid(axis="y")

    if min_ascore is None:
        return

    twinax = ax.twinx()
    _ = twinax.scatter(
        num_suppixs_hist[1:],
        chosen_candidate_mean_ascore_hist,
        label="superpixel mean ascore",
        marker="x",
        color="tab:blue",
    )
    _ = twinax.axhline(min_ascore, color="tab:blue", linestyle="--", label="min ascore")

    _ = twinax.set_ylabel("Superpixel anomaly score")
    _ = twinax.set_yticks(np.linspace(*twinax.get_ylim(), 6))
    _ = twinax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
