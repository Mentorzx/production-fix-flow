const fs = require('fs');
const path = require('path');

// Guardian for the new ESM dashboard:
// - validates that required chart components exist in the source tree
// - does NOT depend on string presence inside the minified bundle
const chartsDir = path.join(__dirname, '../static/js/features/hpo/charts');
const featuresDir = path.join(__dirname, '../static/js/features/hpo');
const allChartsPath = path.join(chartsDir, 'AllCharts.js');
const bundlePath = path.join(__dirname, '../dist/dashboard.js');

const REQUIRED_EXPORTS = [
    'BestTrialCard',
    'LearningCurveChart',
    'ELBOBreakdownCard',
    'PC2MetricsCard',
    'ParamImportanceCard',
    'IncumbentTrajectoryCard',
    'EDFPlotCard',
    'SearchSpaceTableCard',
    'TerminalLogCard',
    'ParetoFrontCard',
    'ParallelCoordinatesCard',
    'CorrelationMatrixCard',
    'StructuralMetricsCard',
    'LatencyParetoCard',
    'ContourPlotCard',
    'InteractionPlotCard',
    'TimelinePlotCard',
    'HypervolumeCard',
    'PCComparisonTableCard',
    'DetailedHistoryCard',
    'FullMetricsLogCard',
    'ScatterPlotCard',
    'MetricsEvolutionCard',
    'HardwareMonitorCard',
    'GradientHealthCard',
    'RawConfigCard',
    'EstimatedScoreCard',
    'OptimizationVelocityCard',
    'LossProjectionCard',
    'EarlyStoppingGaugeCard',
    'ComposedChartCard',
    'ConfusionMatrixCard',
];

function collectFiles(dirPath, out = []) {
    if (!fs.existsSync(dirPath)) return out;
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
    for (const ent of entries) {
        const p = path.join(dirPath, ent.name);
        if (ent.isDirectory()) collectFiles(p, out);
        else out.push(p);
    }
    return out;
}

function hasExportDefinition(content, name) {
    // Covers: export const X = ..., export function X(...), export class X ...
    const re = new RegExp(`export\\s+(?:const|function|class)\\s+${name}\\b`, 'm');
    return re.test(content);
}

console.log('\x1b[34m--- PFF DASHBOARD GUARDIAN (ESM) ---\x1b[0m');
const errors = [];

if (!fs.existsSync(allChartsPath)) {
    errors.push(`[MISSING] ${allChartsPath} (AllCharts export surface not found)`);
}
if (!fs.existsSync(bundlePath)) {
    errors.push(`[MISSING] ${bundlePath} (bundle not built)`);
}
if (!fs.existsSync(chartsDir)) {
    errors.push(`[MISSING] ${chartsDir} (charts directory not found)`);
}

const candidateFiles = [
    ...collectFiles(chartsDir).filter(p => p.endsWith('.js') || p.endsWith('.jsx')),
    ...collectFiles(featuresDir).filter(p => p.endsWith('.js') || p.endsWith('.jsx')),
];

const fileContents = new Map();
for (const fp of candidateFiles) {
    try {
        fileContents.set(fp, fs.readFileSync(fp, 'utf8'));
    } catch {
        // ignore
    }
}

for (const name of REQUIRED_EXPORTS) {
    let found = false;
    for (const content of fileContents.values()) {
        if (hasExportDefinition(content, name)) {
            found = true;
            break;
        }
    }
    if (!found) {
        errors.push(`[SOURCE MISSING] Export '${name}' not found under static/js/features/hpo/**`);
    }
}

if (errors.length > 0) {
    console.error('\x1b[31m\nCRITICAL ERROR: Integridade do Dashboard comprometida!\x1b[0m');
    errors.forEach(err => console.error(` \x1b[41m\x1b[37m FAIL \x1b[0m ${err}`));
    process.exit(1);
}

console.log(`\x1b[32mSUCCESS: ${REQUIRED_EXPORTS.length} exports found in source tree and bundle exists.\x1b[0m`);
process.exit(0);
