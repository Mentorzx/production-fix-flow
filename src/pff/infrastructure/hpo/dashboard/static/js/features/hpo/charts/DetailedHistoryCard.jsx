/**
 * Provide DetailedHistoryCard module functionality for the HPO dashboard.
 */

import { MetricsTableCard } from "./MetricsTableCard.jsx";

/**
 * Expose detailed history card for dashboard usage.
 */
export const DetailedHistoryCard = ({ trials }) => (
  <MetricsTableCard
    title="Ranking de Trials"
    registryKey="detailed_history"
    data={trials}
    type="trial"
  />
);
