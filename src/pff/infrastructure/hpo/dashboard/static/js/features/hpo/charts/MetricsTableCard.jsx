import { Card, TableIcon, EmptyState } from "../../../ui/BaseComponents.jsx";
import { MetricsHistoryTable } from "../MetricsHistoryTable.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

/**
 * Reusable metrics table card shell.
 *
 * Standardizes the Card + MetricsHistoryTable pattern used by both
 * study-level (trial ranking) and trial-level (epoch log) views.
 */
export const MetricsTableCard = ({ title, registryKey, data, type = "trial", compact = false }) => (
  <Card
    title={title}
    icon={TableIcon}
    className="min-h-0 max-h-[65vh]"
    helpText={registryKey ? ChartRegistry.get(registryKey) : undefined}
  >
    {data && data.length > 0 ? (
      <div className="flex-1 min-h-0">
        <MetricsHistoryTable data={data} type={type} compact={compact} framed={false} />
      </div>
    ) : (
      <EmptyState className="text-sm">Sem dados</EmptyState>
    )}
  </Card>
);
