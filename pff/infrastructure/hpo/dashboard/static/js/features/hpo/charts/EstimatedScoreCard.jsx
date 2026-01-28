import { Card, TargetIcon } from "../../../ui/BaseComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const EstimatedScoreCard = ({ projection, totalTrials = 50 }) => (
    <Card title="Estimativa de Score Final" icon={TargetIcon} className="h-full" helpText={ChartRegistry.get('estimated_score')}>
        <div className="flex flex-col items-center justify-center h-full gap-2">
            <div className="text-4xl font-black text-orange-400 font-mono">{projection?.predictedValue?.toFixed(4) ?? "—"}</div>
            <div className="text-[10px] text-zinc-500 uppercase tracking-widest">Alvo para Trial #{totalTrials}</div>
        </div>
    </Card>
);
