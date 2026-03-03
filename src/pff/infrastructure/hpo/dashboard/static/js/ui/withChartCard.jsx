/**
 * Composition wrapper for the canonical chart card shell:
 * Card → ChartFrame → WithData → ChartContainer.
 *
 * Eliminates ~12 LOC of repeated boilerplate per chart card.
 * The chart component passes its children (the Recharts element)
 * and controls the `hasData` guard with access to its computed data.
 *
 * Use for cards that follow the canonical pattern; skip for
 * hybrid layouts (gauges, matrices, tables, big-number cards).
 *
 * @example
 * export const MyCard = ({ trials }) => {
 *   const data = useMemo(() => ..., [trials]);
 *   return (
 *     <ChartCard title="My Chart" icon={Activity} registryKey="myChart"
 *                hasData={data.length > 0} emptyText="Sem dados">
 *       <ComposedChart data={data} margin={defaultChartMargins}>
 *         ...
 *       </ComposedChart>
 *     </ChartCard>
 *   );
 * };
 */
import React from "react";
import { ChartFrame, ChartContainer } from "./ChartPrimitives.jsx";
import { Card } from "./Card.jsx";
import { WithData } from "./EmptyStates.jsx";
import { ChartRegistry } from "../domain/metrics/ChartRegistry.js";

/**
 * @typedef {Object} ChartCardProps
 * @property {string}            title
 * @property {React.FC}          icon
 * @property {string}            registryKey              - Key for ChartRegistry.get()
 * @property {boolean}           hasData                  - Guard: true → render chart, false → show empty state
 * @property {React.ReactNode}   children                 - Single Recharts element (receives width/height from ChartContainer)
 * @property {string}            [emptyText="Sem dados"]
 * @property {string}            [emptyClassName]
 * @property {string}            [chartFrameClassName]
 * @property {number}            [chartContainerMinHeight]
 * @property {string}            [chartContainerClassName]
 * @property {React.ReactNode}   [action]
 * @property {boolean}           [glow]
 * @property {React.ReactNode}   [headerRight]
 */

/** @type {React.FC<ChartCardProps>} */
export const ChartCard = React.memo(
  ({
    title,
    icon,
    registryKey,
    hasData,
    children,
    emptyText = "Sem dados",
    emptyClassName = "",
    chartFrameClassName = "",
    chartContainerMinHeight,
    chartContainerClassName = "",
    action,
    glow,
    headerRight,
  }) => (
    <Card
      title={title}
      icon={icon}
      className="h-full"
      helpText={ChartRegistry.get(registryKey)}
      action={action}
      glow={glow}
      headerRight={headerRight}
    >
      <ChartFrame className={chartFrameClassName}>
        <WithData when={hasData} empty={emptyText} emptyClassName={emptyClassName}>
          <ChartContainer minHeight={chartContainerMinHeight} className={chartContainerClassName}>
            {children}
          </ChartContainer>
        </WithData>
      </ChartFrame>
    </Card>
  )
);

ChartCard.displayName = "ChartCard";
