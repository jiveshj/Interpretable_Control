"""
probe_trainer.py
----------------
Fits one linear probe (Ridge regressor) per layer and evaluates:
  - R²  for exact metric distance   <- main result
  - R²  for log-transformed distance (perception is approx. logarithmic)
  - R²  for bucketed (coarse) distance
  - Accuracy for discrete bin classification

Key design decisions:
  - Ridge (L2-regularized OLS) rather than raw OLS because latent dimensions
    can be highly correlated and dim >> n_samples in shallow layers.
  - Features are standardized (zero-mean, unit-variance) before fitting so
    Ridge's alpha is comparable across layers with different activation scales.
  - Both raw and log-transformed results are reported.
"""

import numpy as np
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.pipeline import Pipeline

from data_utils import split_indices, bucket_distances, N_DISTANCE_BINS


# ── Result containers ────────────────────────────────────────────────────────

@dataclass
class LayerProbeResult:
    layer_name:     str
    layer_dim:      int
    n_train:        int
    n_test:         int

    # Continuous distance regression
    r2_exact:       float    # R² predicting raw distance (m)
    r2_log:         float    # R² predicting log(distance)
    mae_exact:      float    # mean absolute error (m) on test set

    # Coarse (bucketed) metrics
    acc_bins:       float    # accuracy on near/mid/far/very-far bins
    r2_bins:        float    # R² predicting bin index (ordinal)

    # Ridge regularization chosen by grid search
    best_alpha_reg: float
    best_alpha_clf: float

    @property
    def coarse_vs_exact_gap(self) -> float:
        """
        Positive gap = model encodes COARSE distance better than exact metric.
        This is the key metric for the 'rough vs. precise' hypothesis.
        Both R² values are already in [0,1] so they're directly comparable.
        """
        return self.r2_bins - self.r2_exact


@dataclass
class ProbeSweepResults:
    layer_results: list = field(default_factory=list)

    @property
    def layer_names(self) -> list[str]:
        return [r.layer_name for r in self.layer_results]

    @property
    def r2_exact(self) -> np.ndarray:
        return np.array([r.r2_exact for r in self.layer_results])

    @property
    def r2_log(self) -> np.ndarray:
        return np.array([r.r2_log for r in self.layer_results])

    @property
    def r2_bins(self) -> np.ndarray:
        return np.array([r.r2_bins for r in self.layer_results])

    @property
    def acc_bins(self) -> np.ndarray:
        return np.array([r.acc_bins for r in self.layer_results])

    @property
    def coarse_vs_exact_gap(self) -> np.ndarray:
        return np.array([r.coarse_vs_exact_gap for r in self.layer_results])

    def summary_table(self) -> str:
        header = (f"\n  {'Layer':<25} {'Dim':>6}  {'R²(exact)':>10}  "
                  f"{'R²(log)':>8}  {'R²(bins)':>9}  {'Acc(bins)':>10}  "
                  f"{'Gap':>7}  {'MAE(m)':>8}")
        sep = "  " + "-" * 90
        rows = [header, sep]
        for r in self.layer_results:
            rows.append(
                f"  {r.layer_name:<25} {r.layer_dim:>6}  "
                f"{r.r2_exact:>10.3f}  {r.r2_log:>8.3f}  "
                f"{r.r2_bins:>9.3f}  {r.acc_bins:>10.3f}  "
                f"{r.coarse_vs_exact_gap:>+7.3f}  {r.mae_exact:>8.2f}"
            )
        return "\n".join(rows)


# ── Trainer ──────────────────────────────────────────────────────────────────

class LinearProbeTrainer:
    """
    Fits and evaluates linear probes for each layer in a latent dictionary.

    Parameters
    ----------
    train_frac, val_frac : split fractions (test = 1 - train - val)
    use_log_distance     : whether to also fit a log-distance probe
    alpha_grid           : regularization strengths to sweep over
    seed                 : random seed for reproducibility
    """

    def __init__(
        self,
        train_frac:       float = 0.70,
        val_frac:         float = 0.15,
        use_log_distance: bool  = True,
        alpha_grid:       list  = None,
        seed:             int   = 42,
    ):
        self.train_frac       = train_frac
        self.val_frac         = val_frac
        self.use_log_distance = use_log_distance
        self.alpha_grid       = alpha_grid or [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
        self.seed             = seed

    def run(
        self,
        latents:     dict,
        gt_distance: np.ndarray,
        gt_bins:     np.ndarray,
        verbose:     bool = True,
    ) -> ProbeSweepResults:
        """
        Run the full probe sweep over all layers.

        Parameters
        ----------
        latents     : {layer_name: np.ndarray (N, D)}
        gt_distance : (N,) ground-truth distances in meters
        gt_bins     : (N,) discretised bin labels
        """
        n = len(gt_distance)
        idx_tr, idx_va, idx_te = split_indices(n, self.train_frac, self.val_frac, self.seed)

        log_distance = np.log1p(gt_distance)  # log(1 + d) is always non-negative

        results = ProbeSweepResults()

        for layer_name, X in latents.items():
            if verbose:
                print(f"  Probing {layer_name:<25}  dim={X.shape[1]}", end=" ... ", flush=True)

            X_tr, X_va, X_te = X[idx_tr], X[idx_va], X[idx_te]

            y_exact_tr, y_exact_va, y_exact_te = (
                gt_distance[idx_tr], gt_distance[idx_va], gt_distance[idx_te])
            y_log_tr, y_log_va, y_log_te = (
                log_distance[idx_tr], log_distance[idx_va], log_distance[idx_te])
            y_bins_tr, y_bins_va, y_bins_te = (
                gt_bins[idx_tr], gt_bins[idx_va], gt_bins[idx_te])

            # ── Exact distance regression ──────────────────────────────────
            alpha_exact, pipe_exact = self._fit_ridge(
                X_tr, y_exact_tr, X_va, y_exact_va)
            preds_exact = pipe_exact.predict(X_te)
            r2_exact    = float(r2_score(y_exact_te, preds_exact))
            mae_exact   = float(np.mean(np.abs(y_exact_te - preds_exact)))

            # ── Log distance regression ────────────────────────────────────
            r2_log = 0.0
            if self.use_log_distance:
                alpha_log, pipe_log = self._fit_ridge(
                    X_tr, y_log_tr, X_va, y_log_va)
                r2_log = float(r2_score(y_log_te, pipe_log.predict(X_te)))

            # ── Coarse bin ordinal regression ──────────────────────────────
            alpha_bins, pipe_bins = self._fit_ridge(
                X_tr, y_bins_tr.astype(float), X_va, y_bins_va.astype(float))
            preds_bins_cont = pipe_bins.predict(X_te)
            r2_bins = float(r2_score(y_bins_te.astype(float), preds_bins_cont))

            # ── Coarse bin classification (accuracy) ───────────────────────
            alpha_clf, pipe_clf = self._fit_ridge_clf(
                X_tr, y_bins_tr, X_va, y_bins_va)
            acc_bins = float(accuracy_score(y_bins_te, pipe_clf.predict(X_te)))

            result = LayerProbeResult(
                layer_name     = layer_name,
                layer_dim      = X.shape[1],
                n_train        = len(idx_tr),
                n_test         = len(idx_te),
                r2_exact       = r2_exact,
                r2_log         = r2_log,
                mae_exact      = mae_exact,
                acc_bins       = acc_bins,
                r2_bins        = r2_bins,
                best_alpha_reg = alpha_exact,
                best_alpha_clf = alpha_clf,
            )
            results.layer_results.append(result)

            if verbose:
                print(f"R²={r2_exact:.3f}  R²(bins)={r2_bins:.3f}  "
                      f"gap={result.coarse_vs_exact_gap:+.3f}  MAE={mae_exact:.1f}m")

        return results

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _fit_ridge(self, X_tr, y_tr, X_va, y_va):
        """Grid-search alpha on val set, then refit on train+val."""
        best_alpha = self.alpha_grid[0]
        best_r2    = -np.inf
        best_pipe  = None

        for alpha in self.alpha_grid:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge",  Ridge(alpha=alpha, fit_intercept=True, solver = "svd")),
            ])
            pipe.fit(X_tr, y_tr)
            r2 = r2_score(y_va, pipe.predict(X_va))
            if r2 > best_r2:
                best_r2    = r2
                best_alpha = alpha
                best_pipe  = pipe

        X_full = np.concatenate([X_tr, X_va], axis=0)
        y_full = np.concatenate([y_tr, y_va], axis=0)
        best_pipe.fit(X_full, y_full)
        return best_alpha, best_pipe

    def _fit_ridge_clf(self, X_tr, y_tr, X_va, y_va):
        """Ridge classifier with alpha grid search on val set."""
        best_alpha = self.alpha_grid[0]
        best_acc   = -np.inf
        best_pipe  = None

        for alpha in self.alpha_grid:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    RidgeClassifier(alpha=alpha)),
            ])
            pipe.fit(X_tr, y_tr)
            acc = accuracy_score(y_va, pipe.predict(X_va))
            if acc > best_acc:
                best_acc   = acc
                best_alpha = alpha
                best_pipe  = pipe

        X_full = np.concatenate([X_tr, X_va], axis=0)
        y_full = np.concatenate([y_tr, y_va], axis=0)
        best_pipe.fit(X_full, y_full)
        return best_alpha, best_pipe
