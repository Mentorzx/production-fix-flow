export const formatParamValue = (v) => {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return v.toFixed(v < 0.01 ? 6 : 4);
  return String(v);
};

export const formatMetricValue = (val) => {
  if (val === null || val === undefined) return "—";
  return typeof val === "number" ? val.toFixed(4) : String(val);
};

export const formatDuration = (value, isCompact = false) => {
  if (!value) return "—";
  if (isCompact || value < 60) return `${value.toFixed(1)}s`;
  const mins = Math.floor(value / 60);
  return `${mins}m ${(value % 60).toFixed(0)}s`;
};

export const resolveMetricValue = (row, key) => {
  if (!row) return null;
  const mapping = {
    score: ["value", "objective", "metrics.score", "score"],
    epoch: ["epoch", "metrics.epoch"],
    best_mcc: [
      "best_mcc",
      "best_val_mcc",
      "metrics.best_mcc",
      "metrics.best_val_mcc",
      "learner.best_mcc",
      "learner.best_val_mcc",
    ],
    best_mrr: [
      "best_mrr",
      "best_val_mrr",
      "metrics.best_mrr",
      "metrics.best_val_mrr",
      "kge.best_mrr",
      "best_val_mrr",
    ],
    mcc: ["mcc", "metrics.mcc", "learner.mcc"],
    accuracy: ["accuracy", "metrics.accuracy", "learner.accuracy"],
    f1: ["f1", "metrics.f1", "learner.f1"],
    auc: ["auc", "metrics.auc", "metrics.roc_auc", "learner.auc", "learner.roc_auc"],
    pr_auc: ["pr_auc", "metrics.pr_auc", "learner.pr_auc"],
    mrr: ["mrr", "metrics.mrr", "kge.mrr", "best_val_mrr"],
    hits1: ["hits1", "hits@1", "metrics.hits1", "metrics.hits@1", "kge.hits1", "kge.hits@1"],
    hits3: ["hits3", "hits@3", "metrics.hits3", "metrics.hits@3", "kge.hits3", "kge.hits@3"],
    hits10: ["hits10", "hits@10", "metrics.hits10", "metrics.hits@10", "kge.hits10", "kge.hits@10"],
    duration: ["metrics.duration", "duration"],
    efficiency: ["metrics.efficiency", "efficiency"],
    loss: [
      "loss",
      "metrics.loss",
      "val_loss",
      "metrics.val_loss",
      "train_loss",
      "metrics.train_loss",
      "metrics.binary_loss",
      "binary_loss",
    ],
  };

  const candidates = mapping[key] || null;
  const state = String(row.state || "").toUpperCase();

  if (candidates) {
    for (const path of candidates) {
      const val = path.split(".").reduce((obj, k) => obj?.[k], row);
      if (val !== undefined && val !== null) {
        if (typeof val === "object") return null;
        if (
          key !== "duration" &&
          state &&
          state !== "COMPLETE" &&
          typeof val === "number" &&
          val === 0
        ) {
          return null;
        }
        return val;
      }
    }
  }

  if (row[key] !== undefined && row[key] !== null) {
    if (typeof row[key] === "object" || typeof row[key] === "function") return null;
    if (
      key !== "duration" &&
      state &&
      state !== "COMPLETE" &&
      typeof row[key] === "number" &&
      row[key] === 0
    ) {
      return null;
    }
    return row[key];
  }

  return null;
};
