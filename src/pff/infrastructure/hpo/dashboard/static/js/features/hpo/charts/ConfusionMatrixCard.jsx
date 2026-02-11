import { Card, TableIcon, WithData } from "../../../ui/BaseComponents.jsx";
import { ConfusionMatrix } from "../ConfusionMatrix.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

export const ConfusionMatrixCard = ({ liveStatus, bestTrial }) => {
    const cm = bestTrial?.metrics?.confusion_matrix || liveStatus?.confusion_matrix || null;

    return (
        <Card title="Matriz de Confusão" className="h-full" icon={TableIcon} helpText={ChartRegistry.get('confusion_matrix')}>
            <div className="h-full w-full p-3 min-h-0 flex">
                <div className="flex-1 min-h-0">
                    <WithData when={!!cm} empty="Aguardando..." emptyClassName="text-zinc-500">
                        <ConfusionMatrix cm={cm} />
                    </WithData>
                </div>
            </div>
        </Card>
    );
};
