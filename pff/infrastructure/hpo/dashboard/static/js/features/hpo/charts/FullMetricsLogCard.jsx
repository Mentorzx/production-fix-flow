import { Card, TableIcon, EmptyState } from "../../../ui/BaseComponents.jsx";
import { MetricsHistoryTable } from "../MetricsHistoryTable.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const FullMetricsLogCard = ({ liveStatus }) => (
    <Card title="Full Metrics Log" icon={TableIcon} className="h-full min-h-0" helpText={ChartRegistry.get('full_metrics_log')}>
        <div className="h-full min-h-0">
            {liveStatus?.epoch_history ? (
                <MetricsHistoryTable data={liveStatus.epoch_history} type="epoch" />
            ) : (
                <EmptyState className="text-sm">No Data</EmptyState>
            )}
        </div>
    </Card>
);
