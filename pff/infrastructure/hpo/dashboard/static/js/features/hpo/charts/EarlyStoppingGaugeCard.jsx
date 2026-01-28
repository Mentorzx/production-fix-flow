import { useMemo } from 'react';

import { Card, AlertTriangle } from "../../../ui/BaseComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const EarlyStoppingGaugeCard = ({ liveData }) => {
    const prob = useMemo(() => { if (!liveData || liveData.length < 5) return 0.1; const recent = liveData.slice(-5); return (recent[0].loss - recent[recent.length - 1].loss) < 0.001 ? 0.8 : 0.2; }, [liveData]);
    return (
        <Card title="Early Stopping" icon={AlertTriangle} className="h-full" helpText={ChartRegistry.get('early_stopping')}>
            <div className="h-full flex flex-col items-center justify-center bg-zinc-900/30 rounded-lg p-6">
                <div className={`text-6xl font-black font-mono ${prob > 0.5 ? 'text-rose-500' : 'text-lime-500'}`}>{(prob * 100).toFixed(0)}%</div>
                <div className="text-zinc-500 text-[10px] mt-2 uppercase tracking-widest font-black">Probabilidade de Parada</div>
            </div>
        </Card>
    );
};
