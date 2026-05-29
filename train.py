import pickle
import warnings

import numpy as np
import scipy.io as sio
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")


CLIP_VALUE = 130.0
TOP_K = 3
MODEL_PATH = "model.pkl"


def make_light_ordinal_geometry_features(D, BS, bounds, clip_value=130.0, top_k=3):
    eps = 1e-9
    n, m = D.shape

    D_clip = np.clip(D, 0, clip_value)

    x_min, y_min, x_max, y_max = bounds
    corners = np.array([
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_min],
        [x_max, y_max],
    ])

    bs_max = np.zeros(m)
    for i in range(m):
        bs_max[i] = np.max(np.sqrt(np.sum((corners - BS[i]) ** 2, axis=1)))

    excess = np.maximum(0, D - bs_max)
    ratio = D / (bs_max + eps)
    invalid = (D > bs_max).astype(float)

    sorted_idx = np.argsort(D_clip, axis=1)
    ranks = np.zeros_like(sorted_idx, dtype=float)

    for i in range(n):
        ranks[i, sorted_idx[i]] = np.arange(1, m + 1)

    top_features = []
    for k in range(top_k):
        onehot = np.zeros((n, m))
        idx = sorted_idx[:, k]
        onehot[np.arange(n), idx] = 1.0
        top_features.append(onehot)

    sorted_d = np.take_along_axis(D_clip, sorted_idx, axis=1)

    gap12 = (sorted_d[:, 1] - sorted_d[:, 0]).reshape(-1, 1)
    gap23 = (sorted_d[:, 2] - sorted_d[:, 1]).reshape(-1, 1)
    gap13 = (sorted_d[:, 2] - sorted_d[:, 0]).reshape(-1, 1)

    gap_features = np.hstack([
        gap12,
        gap23,
        gap13,
    ])

    pair_order = []
    for a in range(m):
        for b in range(a + 1, m):
            order_ab = (D_clip[:, a] < D_clip[:, b]).astype(float).reshape(-1, 1)
            pair_order.append(order_ab)

    pair_order = np.hstack(pair_order)

    stats = np.column_stack([
        np.mean(D_clip, axis=1),
        np.std(D_clip, axis=1),
        np.min(D_clip, axis=1),
        np.max(D_clip, axis=1),
        np.median(D_clip, axis=1),
    ])

    X = np.hstack([
        D_clip,
        stats,
        excess,
        ratio,
        invalid,
        ranks,
        *top_features,
        gap_features,
        pair_order,
    ])

    return X


def make_boundary_aware_features(D, BS, bounds, clip_value=130.0, top_k=3):
    eps = 1e-9
    n, m = D.shape

    base_feat = make_light_ordinal_geometry_features(
        D,
        BS,
        bounds=bounds,
        clip_value=clip_value,
        top_k=top_k,
    )

    D_clip = np.clip(D, 0, clip_value)
    sorted_idx = np.argsort(D_clip, axis=1)
    sorted_d = np.take_along_axis(D_clip, sorted_idx, axis=1)

    bs_x = BS[:, 0]
    bs_y = BS[:, 1]

    x_med = np.median(bs_x)
    y_med = np.median(bs_y)

    left_mask = bs_x <= x_med
    right_mask = bs_x > x_med
    lower_mask = bs_y <= y_med
    upper_mask = bs_y > y_med

    def group_mean(mask):
        if np.sum(mask) == 0:
            return np.zeros((n, 1))
        return np.mean(D_clip[:, mask], axis=1, keepdims=True)

    def group_min(mask):
        if np.sum(mask) == 0:
            return np.zeros((n, 1))
        return np.min(D_clip[:, mask], axis=1, keepdims=True)

    left_mean = group_mean(left_mask)
    right_mean = group_mean(right_mask)
    lower_mean = group_mean(lower_mask)
    upper_mean = group_mean(upper_mask)

    left_min = group_min(left_mask)
    right_min = group_min(right_mask)
    lower_min = group_min(lower_mask)
    upper_min = group_min(upper_mask)

    directional_group_feat = np.hstack([
        left_mean,
        right_mean,
        lower_mean,
        upper_mean,
        left_min,
        right_min,
        lower_min,
        upper_min,
        right_mean - left_mean,
        upper_mean - lower_mean,
        right_min - left_min,
        upper_min - lower_min,
    ])

    d_min = np.min(D_clip, axis=1, keepdims=True)
    d_max = np.max(D_clip, axis=1, keepdims=True)
    d_mean = np.mean(D_clip, axis=1, keepdims=True)
    d_std = np.std(D_clip, axis=1, keepdims=True)

    spread = d_max - d_min
    max_min_ratio = d_max / (d_min + eps)
    std_mean_ratio = d_std / (d_mean + eps)

    top1_top5_gap = (sorted_d[:, 4] - sorted_d[:, 0]).reshape(-1, 1)
    top1_top10_gap = (sorted_d[:, 9] - sorted_d[:, 0]).reshape(-1, 1)

    boundary_score_feat = np.hstack([
        spread,
        max_min_ratio,
        std_mean_ratio,
        top1_top5_gap,
        top1_top10_gap,
    ])

    top3_idx = sorted_idx[:, :3]
    top5_idx = sorted_idx[:, :5]

    top3_centroids = np.zeros((n, 2))
    top5_centroids = np.zeros((n, 2))

    for i in range(n):
        top3_centroids[i] = np.mean(BS[top3_idx[i]], axis=0)
        top5_centroids[i] = np.mean(BS[top5_idx[i]], axis=0)

    centroid_feat = np.hstack([
        top3_centroids,
        top5_centroids,
        top3_centroids - top5_centroids,
    ])


    bs_center = np.mean(BS, axis=0)
    anchor_vec = BS - bs_center

    inv_d = 1.0 / (D_clip + eps)
    inv_d = inv_d / (np.sum(inv_d, axis=1, keepdims=True) + eps)

    direction_vec = inv_d @ anchor_vec
    direction_norm = np.sqrt(np.sum(direction_vec ** 2, axis=1, keepdims=True))

    direction_feat = np.hstack([
        direction_vec,
        direction_norm,
    ])



    virtual_top3 = top3_centroids
    virtual_top5 = top5_centroids

    virtual_mirror = 2.0 * bs_center - top3_centroids

    d_virtual_top3 = np.mean(sorted_d[:, :3], axis=1, keepdims=True)
    d_virtual_top5 = np.mean(sorted_d[:, :5], axis=1, keepdims=True)

    d_virtual_mirror = d_virtual_top3 + 0.5 * spread

    virtual_top3_vec = virtual_top3 - bs_center
    virtual_top5_vec = virtual_top5 - bs_center
    virtual_mirror_vec = virtual_mirror - bs_center

    virtual_top3_norm = np.sqrt(np.sum(virtual_top3_vec ** 2, axis=1, keepdims=True))
    virtual_top5_norm = np.sqrt(np.sum(virtual_top5_vec ** 2, axis=1, keepdims=True))
    virtual_mirror_norm = np.sqrt(np.sum(virtual_mirror_vec ** 2, axis=1, keepdims=True))

    mirror_gap_vec = virtual_mirror - virtual_top3
    mirror_gap_norm = np.sqrt(np.sum(mirror_gap_vec ** 2, axis=1, keepdims=True))

    virtual_anchor_feat = np.hstack([
        virtual_top3,
        virtual_top5,
        virtual_mirror,
        d_virtual_top3,
        d_virtual_top5,
        d_virtual_mirror,
        virtual_top3_vec,
        virtual_top5_vec,
        virtual_mirror_vec,
        virtual_top3_norm,
        virtual_top5_norm,
        virtual_mirror_norm,
        mirror_gap_vec,
        mirror_gap_norm,
    ])

    X = np.hstack([
        base_feat,
        directional_group_feat,
        boundary_score_feat,
        centroid_feat,
        direction_feat,
        virtual_anchor_feat,
    ])

    return X


def evaluate(pred, true):
    err = np.sqrt(np.sum((pred - true) ** 2, axis=1))

    return {
        "MAE": float(np.mean(err)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "P75": float(np.percentile(err, 75)),
        "P90": float(np.percentile(err, 90)),
        "MAX": float(np.max(err)),
    }


def main():
    mat_path = "DH_FR1.mat"

    data = sio.loadmat(mat_path, squeeze_me=False)

    BS_positions = np.asarray(data["BS_positions"], dtype=float)
    d_hat = np.asarray(data["d_hat"], dtype=float)
    p = np.asarray(data["p"], dtype=float)

    BS = BS_positions.T
    D = d_hat.T     
    Y = p.T      

    bounds = (
        float(np.min(Y[:, 0])),
        float(np.min(Y[:, 1])),
        float(np.max(Y[:, 0])),
        float(np.max(Y[:, 1])),
    )

    X = make_boundary_aware_features(
        D,
        BS,
        bounds=bounds,
        clip_value=CLIP_VALUE,
        top_k=TOP_K,
    )

    params = dict(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.07,
        min_samples_leaf=3,
        subsample=0.85,
        random_state=42,
    )

    model_x = GradientBoostingRegressor(**params)
    model_y = GradientBoostingRegressor(**params)

    model_x.fit(X, Y[:, 0])
    model_y.fit(X, Y[:, 1])

    pred = np.column_stack([
        model_x.predict(X),
        model_y.predict(X),
    ])

    metrics = evaluate(pred, Y)

    print("Training feature shape:", X.shape)
    print("Training metrics:", metrics)

    bundle = {
        "model_x": model_x,
        "model_y": model_y,
        "bounds": bounds,
        "clip_value": CLIP_VALUE,
        "top_k": TOP_K,
        "algorithm": "Boundary Ranking GBM with Virtual Anchor Feature",
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)


if __name__ == "__main__":
    main()