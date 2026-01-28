import { ComposedChart } from 'recharts';

import { Card, TrendingUp, Search, colors, defaultChartMargins, ChartFrame, ChartContainer, WithData } from "../../../ui/BaseComponents.jsx";

export const ComposedChartCard = ({ title, icon, data, helpText, children, hasData }) => {
    const IconComp = icon === 'TrendingUp' ? TrendingUp : Search;
    return (
        <Card title={title} icon={IconComp} className="h-full" glow helpText={helpText}>
            <ChartFrame>
                <WithData when={hasData} empty="Dados insuficientes">
                    <ChartContainer>
                        <ComposedChart data={data} margin={defaultChartMargins}>{children({ data, colors })}</ComposedChart>
                    </ChartContainer>
                </WithData>
            </ChartFrame>
        </Card>
    );
};
