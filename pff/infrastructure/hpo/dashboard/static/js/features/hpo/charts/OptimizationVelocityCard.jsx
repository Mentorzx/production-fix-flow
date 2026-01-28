import { Card, Zap } from "../../../ui/BaseComponents.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const OptimizationVelocityCard = ({ projection }) => (
    <Card title="Velocidade de Otimização" icon={Zap} className="h-full" helpText={ChartRegistry.get('optimization_velocity')}>
        <div className="flex flex-col h-full min-h-0">
            <div className="flex-1 min-h-0 flex items-center justify-center">
                <div className="text-center">
                    <div className={`text-3xl font-black font-mono tabular-nums ${projection?.slope > 0 ? 'text-lime-400' : 'text-rose-400'}`}>
                        {projection?.slope ? (projection.slope * 100).toFixed(6) + '%' : "—"}
                    </div>
                    <div className="mt-2 text-[10px] text-zinc-500 uppercase tracking-widest">Inclinação por trial</div>
                </div>
            </div>
        </div>
    </Card>
);
