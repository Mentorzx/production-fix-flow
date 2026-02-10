import { MetricsTableCard } from "./MetricsTableCard.jsx";

export const FullMetricsLogCard = ({ liveStatus }) => (
    <MetricsTableCard
        title="Log de Métricas"
        registryKey="full_metrics_log"
        data={liveStatus?.epoch_history}
        type="epoch"
    />
);
