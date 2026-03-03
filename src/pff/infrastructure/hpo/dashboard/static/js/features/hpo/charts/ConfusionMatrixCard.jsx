/**
 * Provide ConfusionMatrixCard module functionality for the HPO dashboard.
 */

import { TableIcon } from "../../../ui/icons.jsx";
import { Card } from "../../../ui/Card.jsx";
import { WithData } from "../../../ui/EmptyStates.jsx";
import { ConfusionMatrix } from "../ConfusionMatrix.jsx";
import { ChartRegistry } from "../../../domain/metrics/ChartRegistry.js";

/**
 * Expose confusion matrix card for dashboard usage.
 */
export const ConfusionMatrixCard = ({ liveStatus, bestTrial }) => {
  const cm = bestTrial?.metrics?.confusion_matrix || liveStatus?.confusion_matrix || null;

  return (
    <Card
      title="Matriz de Confusão"
      className="h-full"
      icon={TableIcon}
      helpText={ChartRegistry.get("confusion_matrix")}
    >
      <div
        className="h-full w-full p-3 min-h-0 flex border rounded-xl items-center justify-center"
        style={{
          borderColor: "var(--viz-border)",
          background:
            "linear-gradient(160deg, color-mix(in srgb, var(--viz-bg-surface), transparent 2%) 0%, color-mix(in srgb, var(--viz-bg-canvas), transparent 16%) 100%)",
        }}
      >
        <div className="flex-1 min-h-0 w-full">
          <WithData when={!!cm} empty="Aguardando..." emptyClassName="text-zinc-500">
            <ConfusionMatrix cm={cm} />
          </WithData>
        </div>
      </div>
    </Card>
  );
};
