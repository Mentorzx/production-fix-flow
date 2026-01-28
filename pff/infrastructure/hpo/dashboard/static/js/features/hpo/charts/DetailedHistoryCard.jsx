import { Card, TableIcon } from "../../../ui/BaseComponents.jsx";
import { MetricsHistoryTable } from "../MetricsHistoryTable.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const DetailedHistoryCard = ({ trials }) => (
    <Card title="Ranking de Trials" icon={TableIcon} className="min-h-[400px]" helpText={ChartRegistry.get('detailed_history')}>
        <MetricsHistoryTable data={trials} />
    </Card>
);
