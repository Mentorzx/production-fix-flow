#!/usr/bin/env python3
"""
Real-time Alert System
WebSocket-based real-time monitoring and alerting for pipeline health.

Features:
1. Real-time log monitoring
2. WebSocket server for live alerts
3. Dashboard integration
4. Alert severity levels and escalation
5. Historical alert tracking
"""

import asyncio
import websockets
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import asyncio
from websockets.server import WebSocketServerProtocol
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/realtime_alerts.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Real-time alert data structure."""
    id: str
    severity: str  # CRITICAL, WARNING, INFO
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    escalation_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.acknowledged_at:
            data['acknowledged_at'] = self.acknowledged_at.isoformat()
        return data

@dataclass
class MetricThreshold:
    """Threshold configuration for metrics."""
    name: str
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    severity: str = "WARNING"
    escalation_enabled: bool = True
    check_interval: int = 30  # seconds

class RealtimeAlertSystem:
    """Real-time alert monitoring system."""

    def __init__(self, config_path: str = "config/alerts.yaml"):
        self.config_path = Path(config_path)
        self.clients: Set[WebSocketServerProtocol] = set()
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.thresholds: Dict[str, MetricThreshold] = {}
        self.active_monitors: Dict[str, threading.Thread] = {}
        self.monitoring_active = False
        self.alert_counter = 0

        # Load configuration
        self.load_config()

        # Initialize default thresholds
        self.init_default_thresholds()

    def load_config(self):
        """Load alert configuration from YAML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # Load thresholds from config
                if 'thresholds' in config:
                    for name, thresh_config in config['thresholds'].items():
                        self.thresholds[name] = MetricThreshold(
                            name=name,
                            min_threshold=thresh_config.get('min_threshold'),
                            max_threshold=thresh_config.get('max_threshold'),
                            severity=thresh_config.get('severity', 'WARNING'),
                            escalation_enabled=thresh_config.get('escalation_enabled', True),
                            check_interval=thresh_config.get('check_interval', 30)
                        )

                logger.info(f"✅ Loaded {len(self.thresholds)} thresholds from config")
            else:
                logger.info("📝 No config file found, using defaults")

        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")

    def init_default_thresholds(self):
        """Initialize default metric thresholds."""
        defaults = {
            'violation_percentage': MetricThreshold(
                name='violation_percentage',
                max_threshold=200.0,
                severity='CRITICAL',
                check_interval=30
            ),
            'symbolic_contribution': MetricThreshold(
                name='symbolic_contribution',
                max_threshold=85.0,
                severity='CRITICAL',
                check_interval=60
            ),
            'hybrid_contribution': MetricThreshold(
                name='hybrid_contribution',
                min_threshold=15.0,
                severity='WARNING',
                check_interval=60
            ),
            'feature_324_importance': MetricThreshold(
                name='feature_324_importance',
                min_threshold=0.01,
                severity='WARNING',
                check_interval=120
            ),
            'f1_score': MetricThreshold(
                name='f1_score',
                min_threshold=0.65,
                severity='WARNING',
                check_interval=60
            ),
            'numba_success_rate': MetricThreshold(
                name='numba_success_rate',
                min_threshold=0.95,
                severity='WARNING',
                check_interval=30
            ),
            'error_count': MetricThreshold(
                name='error_count',
                max_threshold=5.0,
                severity='CRITICAL',
                check_interval=10
            ),
            'processing_time': MetricThreshold(
                name='processing_time',
                max_threshold=120.0,
                severity='WARNING',
                check_interval=60
            )
        }

        # Add defaults for any missing thresholds
        for name, threshold in defaults.items():
            if name not in self.thresholds:
                self.thresholds[name] = threshold

        logger.info(f"📊 Initialized {len(self.thresholds)} metric thresholds")

    async def register_client(self, websocket: WebSocketServerProtocol):
        """Register a new WebSocket client."""
        self.clients.add(websocket)
        logger.info(f"🔗 Client connected: {websocket.remote_address}")

        # Send current status
        await self.send_current_status(websocket)

    async def unregister_client(self, websocket: WebSocketServerProtocol):
        """Unregister a WebSocket client."""
        self.clients.discard(websocket)
        logger.info(f"🔌 Client disconnected: {websocket.remote_address}")

    async def send_current_status(self, websocket: WebSocketServerProtocol):
        """Send current status to a client."""
        try:
            status = {
                'type': 'status_update',
                'data': {
                    'active_alerts': len([a for a in self.alerts.values() if not a.resolved]),
                    'total_alerts': len(self.alerts),
                    'monitoring_active': self.monitoring_active,
                    'connected_clients': len(self.clients),
                    'recent_metrics': self.get_recent_metrics()
                }
            }
            await websocket.send(json.dumps(status))

            # Send recent alerts
            recent_alerts = list(self.alert_history)[-10:]
            for alert in recent_alerts:
                await websocket.send(json.dumps({
                    'type': 'alert',
                    'data': alert.to_dict()
                }))

        except Exception as e:
            logger.error(f"❌ Error sending status to client: {e}")

    def get_recent_metrics(self) -> Dict[str, Any]:
        """Get recent metric values."""
        recent = {}
        for name, history in self.metrics_history.items():
            if history:
                recent[name] = {
                    'current': history[-1],
                    'count': len(history),
                    'avg': sum(history) / len(history)
                }
        return recent

    async def broadcast_alert(self, alert: Alert):
        """Broadcast alert to all connected clients."""
        message = {
            'type': 'alert',
            'data': alert.to_dict()
        }

        # Send to all clients
        disconnected_clients = set()
        for client in self.clients.copy():
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"❌ Error broadcasting alert to client: {e}")
                disconnected_clients.add(client)

        # Remove disconnected clients
        self.clients -= disconnected_clients

        # Log the alert
        severity_emoji = {"CRITICAL": "🔴", "WARNING": "⚠️", "INFO": "ℹ️"}
        logger.warning(f"{severity_emoji.get(alert.severity, '❓')} ALERT: {alert.title} - {alert.message}")

        # Store in history
        self.alert_history.append(alert)

    async def broadcast_metric_update(self, metric_name: str, value: float):
        """Broadcast metric update to all clients."""
        message = {
            'type': 'metric_update',
            'data': {
                'name': metric_name,
                'value': value,
                'timestamp': datetime.now().isoformat()
            }
        }

        disconnected_clients = set()
        for client in self.clients.copy():
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"❌ Error broadcasting metric update: {e}")
                disconnected_clients.add(client)

        self.clients -= disconnected_clients

    def create_alert(self, metric_name: str, current_value: float, threshold: MetricThreshold) -> Alert:
        """Create a new alert."""
        self.alert_counter += 1
        alert_id = f"alert_{self.alert_counter:04d}_{int(time.time())}"

        # Determine threshold direction
        if threshold.min_threshold is not None and current_value < threshold.min_threshold:
            threshold_value = threshold.min_threshold
            message = f"{metric_name} too low: {current_value:.4f} (min: {threshold_value})"
        elif threshold.max_threshold is not None and current_value > threshold.max_threshold:
            threshold_value = threshold.max_threshold
            message = f"{metric_name} too high: {current_value:.4f} (max: {threshold_value})"
        else:
            return None

        # Create alert
        alert = Alert(
            id=alert_id,
            severity=threshold.severity,
            title=f"{metric_name} Alert",
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold_value,
            timestamp=datetime.now()
        )

        # Store and return
        self.alerts[alert_id] = alert
        return alert

    async def check_metric_thresholds(self, metric_name: str, value: float):
        """Check if metric value exceeds thresholds and create alerts."""
        if metric_name not in self.thresholds:
            return

        threshold = self.thresholds[metric_name]

        # Store in history
        self.metrics_history[metric_name].append(value)

        # Check thresholds
        alert = self.create_alert(metric_name, value, threshold)
        if alert:
            await self.broadcast_alert(alert)

    async def monitor_pipeline_logs(self):
        """Monitor pipeline log files for real-time metrics."""
        log_files = list(Path("logs").glob("*.log"))
        if not log_files:
            logger.warning("⚠️ No log files found for monitoring")
            return

        # Get the most recent log file
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"📄 Monitoring log file: {latest_log}")

        last_position = 0

        while self.monitoring_active:
            try:
                # Check if file has grown
                current_size = latest_log.stat().st_size
                if current_size > last_position:
                    # Read new lines
                    with open(latest_log, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()

                    for line in new_lines:
                        await self.process_log_line(line.strip())

                    last_position = current_size

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"❌ Error monitoring logs: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def process_log_line(self, line: str):
        """Process a single log line and extract metrics."""
        import re

        # Define regex patterns for metrics
        patterns = {
            'violation_percentage': r'violations.*?(\d+\.?\d*)%',
            'symbolic_contribution': r'Contribuição das regras simbólicas: (\d+\.?\d*)%',
            'hybrid_contribution': r'Contribuição do modelo híbrido: (\d+\.?\d*)%',
            'f1_score': r'F1-Score Final: (\d+\.?\d*)',
            'numba_success_rate': r'Numba:.*succeeded|Using Numba JIT',
            'error_count': r'ERROR|CRITICAL',
            'processing_time': r'Processed.*samples.*?([\d.]+)s'
        }

        for metric_name, pattern in patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    if metric_name == 'numba_success_rate':
                        # Special handling for success rate
                        value = 1.0 if 'succeeded' in line.lower() else 0.0
                    elif metric_name == 'error_count':
                        # Count errors in the line
                        value = len(re.findall(r'ERROR|CRITICAL', line, re.IGNORECASE))
                    else:
                        value = float(match.group(1))

                    await self.check_metric_thresholds(metric_name, value)
                    await self.broadcast_metric_update(metric_name, value)

                except (ValueError, IndexError) as e:
                    logger.debug(f"Could not parse {metric_name} from line: {line}")

    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            logger.warning("⚠️ Monitoring already active")
            return

        self.monitoring_active = True
        logger.info("🚀 Starting real-time monitoring...")

        # Start log monitoring thread
        log_monitor = threading.Thread(target=self._run_log_monitoring, daemon=True)
        log_monitor.start()
        self.active_monitors['logs'] = log_monitor

        # Start metric checking threads
        for metric_name, threshold in self.thresholds.items():
            if threshold.check_interval > 0:
                monitor = threading.Thread(
                    target=self._run_metric_monitor,
                    args=(metric_name, threshold),
                    daemon=True
                )
                monitor.start()
                self.active_monitors[metric_name] = monitor

    def _run_log_monitoring(self):
        """Run log monitoring in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.monitor_pipeline_logs())

    def _run_metric_monitor(self, metric_name: str, threshold: MetricThreshold):
        """Run metric monitoring in a separate thread."""
        while self.monitoring_active:
            try:
                # Simulate metric checking (in real implementation, this would
                # connect to actual metric sources)
                time.sleep(threshold.check_interval)
            except Exception as e:
                logger.error(f"❌ Error in metric monitor for {metric_name}: {e}")

    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        logger.info("🛑 Stopping real-time monitoring...")

    async def handle_client_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle incoming messages from clients."""
        try:
            data = json.loads(message)
            message_type = data.get('type')

            if message_type == 'acknowledge_alert':
                alert_id = data.get('alert_id')
                if alert_id in self.alerts:
                    self.alerts[alert_id].acknowledged_by = "client"
                    self.alerts[alert_id].acknowledged_at = datetime.now()
                    self.alerts[alert_id].resolved = True

                    # Broadcast acknowledgment
                    await self.broadcast_alert(self.alerts[alert_id])

            elif message_type == 'get_alert_history':
                # Send alert history
                history = [alert.to_dict() for alert in self.alert_history]
                await websocket.send(json.dumps({
                    'type': 'alert_history',
                    'data': history
                }))

            elif message_type == 'get_status':
                # Send current status
                await self.send_current_status(websocket)

        except Exception as e:
            logger.error(f"❌ Error handling client message: {e}")

    async def websocket_handler(self, websocket: WebSocketServerProtocol, path: str):
        """Handle WebSocket connections."""
        await self.register_client(websocket)

        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)

    def create_dashboard_html(self) -> str:
        """Create HTML dashboard for real-time monitoring."""
        dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Health Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .metric-name { color: #666; font-size: 14px; }
        .alert { padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid; }
        .alert.critical { background: #ffe6e6; border-color: #d32f2f; }
        .alert.warning { background: #fff3cd; border-color: #f57c00; }
        .alert.info { background: #e3f2fd; border-color: #1976d2; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-healthy { background: #4caf50; }
        .status-warning { background: #ff9800; }
        .status-critical { background: #f44336; }
        .connection-status { position: fixed; top: 20px; right: 20px; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <div class="connection-status">
        <span class="status-indicator" id="connectionIndicator"></span>
        <span id="connectionText">Disconnected</span>
    </div>

    <div class="header">
        <h1>🏥 Pipeline Health Dashboard</h1>
        <p>Real-time monitoring and alerting system</p>
    </div>

    <div class="metrics-grid" id="metricsGrid">
        <!-- Metrics will be populated here -->
    </div>

    <div>
        <h2>🚨 Recent Alerts</h2>
        <div id="alertsContainer">
            <!-- Alerts will be populated here -->
        </div>
    </div>

    <script>
        class Dashboard {
            constructor() {
                this.ws = null;
                this.metrics = {};
                this.alerts = [];
                this.connect();
            }

            connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.hostname}:8765`;

                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => {
                    this.updateConnectionStatus(true);
                    console.log('Connected to alert system');
                };

                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };

                this.ws.onclose = () => {
                    this.updateConnectionStatus(false);
                    console.log('Disconnected from alert system');
                    // Try to reconnect after 5 seconds
                    setTimeout(() => this.connect(), 5000);
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus(false);
                };
            }

            updateConnectionStatus(connected) {
                const indicator = document.getElementById('connectionIndicator');
                const text = document.getElementById('connectionText');

                if (connected) {
                    indicator.className = 'status-indicator status-healthy';
                    text.textContent = 'Connected';
                } else {
                    indicator.className = 'status-indicator status-critical';
                    text.textContent = 'Disconnected';
                }
            }

            handleMessage(data) {
                switch(data.type) {
                    case 'alert':
                        this.addAlert(data.data);
                        break;
                    case 'metric_update':
                        this.updateMetric(data.data.name, data.data.value);
                        break;
                    case 'status_update':
                        this.updateStatus(data.data);
                        break;
                }
            }

            updateMetric(name, value) {
                this.metrics[name] = value;
                this.renderMetrics();
            }

            renderMetrics() {
                const container = document.getElementById('metricsGrid');
                container.innerHTML = '';

                const metricNames = [
                    'violation_percentage',
                    'symbolic_contribution',
                    'hybrid_contribution',
                    'f1_score',
                    'feature_324_importance',
                    'numba_success_rate'
                ];

                metricNames.forEach(name => {
                    const value = this.metrics[name];
                    const card = this.createMetricCard(name, value);
                    container.appendChild(card);
                });
            }

            createMetricCard(name, value) {
                const card = document.createElement('div');
                card.className = 'metric-card';

                const displayName = name.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());

                let status = 'healthy';
                let valueColor = '#4caf50';

                if (name === 'violation_percentage' && value > 200) {
                    status = 'critical';
                    valueColor = '#f44336';
                } else if (name === 'symbolic_contribution' && value > 85) {
                    status = 'critical';
                    valueColor = '#f44336';
                } else if (name === 'hybrid_contribution' && value < 15) {
                    status = 'warning';
                    valueColor = '#ff9800';
                }

                card.innerHTML = `
                    <div class="metric-name">${displayName}</div>
                    <div class="metric-value" style="color: ${valueColor}">
                        ${value !== undefined ? value.toFixed(4) : 'N/A'}
                    </div>
                    <div class="status-indicator status-${status}"></div>
                `;

                return card;
            }

            addAlert(alert) {
                this.alerts.unshift(alert);
                if (this.alerts.length > 20) {
                    this.alerts.pop();
                }
                this.renderAlerts();
            }

            renderAlerts() {
                const container = document.getElementById('alertsContainer');
                container.innerHTML = '';

                this.alerts.forEach(alert => {
                    const alertEl = document.createElement('div');
                    alertEl.className = `alert ${alert.severity.toLowerCase()}`;

                    const time = new Date(alert.timestamp).toLocaleString();
                    alertEl.innerHTML = `
                        <strong>${alert.title}</strong>
                        <p>${alert.message}</p>
                        <small>${time}</small>
                    `;

                    container.appendChild(alertEl);
                });
            }

            updateStatus(status) {
                // Update overall status indicator
                document.title = `Pipeline Health - ${status.active_alerts > 0 ? '⚠️ Issues' : '✅ Healthy'}`;
            }
        }

        // Initialize dashboard
        const dashboard = new Dashboard();
    </script>
</body>
</html>
        """
        return dashboard_html

async def main():
    """Main execution function."""
    print("🚨 Real-time Alert System")
    print("WebSocket-based real-time monitoring and alerting")
    print("=" * 60)

    # Create alert system
    alert_system = RealtimeAlertSystem()

    # Create dashboard HTML
    dashboard_path = Path("outputs/realtime_dashboard.html")
    dashboard_path.parent.mkdir(exist_ok=True)

    with open(dashboard_path, 'w') as f:
        f.write(alert_system.create_dashboard_html())

    print(f"📊 Dashboard created: {dashboard_path}")
    print("🔗 Open the dashboard file in your browser to view real-time alerts")

    # Start monitoring
    alert_system.start_monitoring()

    # Start WebSocket server
    async with websockets.serve(alert_system.websocket_handler, "localhost", 8765):
        print("🌐 WebSocket server started on ws://localhost:8765")
        print("📡 Monitoring pipeline logs and broadcasting alerts...")
        print("Press Ctrl+C to stop")

        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            print("\\n🛑 Shutting down alert system...")
            alert_system.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())