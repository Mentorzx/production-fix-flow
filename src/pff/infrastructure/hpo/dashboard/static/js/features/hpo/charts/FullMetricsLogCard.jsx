/**
 * Provide FullMetricsLogCard module functionality for the HPO dashboard.
 */

import { MetricsTableCard } from "./MetricsTableCard.jsx";

/**
 * Expose full metrics log card for dashboard usage.
 */
export const FullMetricsLogCard = ({ liveStatus }) => (
  <MetricsTableCard
    title="Log de Métricas"
    registryKey="full_metrics_log"
    data={liveStatus?.epoch_history}
    type="epoch"
  />
);
