/**
 * Centralized statistical functions for the HPO Dashboard.
 * Eliminates duplication of linear regression and Pearson correlation
 * across ForecastTab, RegressionChartCard, RegressionInsightsCard,
 * CorrelationMatrixCard, and PCComparisonTableCard.
 */

/**
 * Ordinary Least Squares linear regression.
 * @param {Array<{x: number, y: number}>} points - Data points.
 * @returns {{slope: number, intercept: number, r2: number, rmse: number, ssTot: number, ssRes: number}}
 */
export const linearRegression = (points) => {
  const n = points.length;
  if (n < 2) return { slope: 0, intercept: 0, r2: 0, rmse: 0, ssTot: 0, ssRes: 0 };

  let sumX = 0,
    sumY = 0,
    sumXY = 0,
    sumXX = 0,
    sumYY = 0;
  for (let i = 0; i < n; i++) {
    const { x, y } = points[i];
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumXX += x * x;
    sumYY += y * y;
  }

  const denom = n * sumXX - sumX * sumX;
  const slope = denom === 0 ? 0 : (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;

  const ssTot = sumYY - (sumY * sumY) / n;
  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    const pred = slope * points[i].x + intercept;
    ssRes += (points[i].y - pred) ** 2;
  }

  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  const rmse = n > 0 ? Math.sqrt(ssRes / n) : 0;

  return { slope, intercept, r2, rmse, ssTot, ssRes };
};

/**
 * Pearson product-moment correlation coefficient.
 * @param {number[]} x - First variable.
 * @param {number[]} y - Second variable.
 * @returns {number} Correlation coefficient in [-1, 1], or 0 if insufficient data.
 */
export const pearsonCorrelation = (x, y) => {
  const n = x.length;
  if (n < 2) return 0;

  let sumX = 0,
    sumY = 0,
    sumXY = 0,
    sumX2 = 0,
    sumY2 = 0;
  for (let i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
    sumXY += x[i] * y[i];
    sumX2 += x[i] * x[i];
    sumY2 += y[i] * y[i];
  }

  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
  return denominator === 0 ? 0 : numerator / denominator;
};

/**
 * Paired Pearson correlation that filters out non-numeric pairs.
 * Used by CorrelationMatrixCard where data may contain nulls.
 * @param {Array} v1 - First variable (may contain non-numbers).
 * @param {Array} v2 - Second variable (may contain non-numbers).
 * @returns {number} Correlation coefficient, or 0 if insufficient numeric pairs.
 */
export const pearsonCorrelationFiltered = (v1, v2) => {
  let sum1 = 0,
    sum2 = 0,
    sum1Sq = 0,
    sum2Sq = 0,
    pSum = 0,
    count = 0;
  for (let i = 0; i < v1.length; i++) {
    if (typeof v1[i] === "number" && typeof v2[i] === "number") {
      sum1 += v1[i];
      sum2 += v2[i];
      sum1Sq += v1[i] * v1[i];
      sum2Sq += v2[i] * v2[i];
      pSum += v1[i] * v2[i];
      count++;
    }
  }
  if (count < 2) return 0;
  const num = pSum - (sum1 * sum2) / count;
  const den = Math.sqrt((sum1Sq - (sum1 * sum1) / count) * (sum2Sq - (sum2 * sum2) / count));
  return den === 0 ? 0 : num / den;
};
