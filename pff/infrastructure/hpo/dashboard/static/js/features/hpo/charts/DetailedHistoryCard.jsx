import { MetricsTableCard } from "./MetricsTableCard.jsx";

export const DetailedHistoryCard = ({ trials }) => (
    <MetricsTableCard title="Ranking de Trials" registryKey="detailed_history" data={trials} type="trial" />
);
