import ReactDOM from 'react-dom/client';
import { StoreProvider } from "./store/store.jsx";
import { ThemeProvider } from "./ui/ThemeContext.jsx";
import { Dashboard } from "./layout/Dashboard.jsx";

console.log("[Main] Bootstrapping Dashboard (Standalone Bundle)...");

const rootElement = document.getElementById("root");
if (!rootElement) {
    console.error("[Main] Root element not found!");
} else {
    const root = ReactDOM.createRoot(rootElement);
    root.render(
        <StoreProvider>
            <ThemeProvider>
                <Dashboard />
            </ThemeProvider>
        </StoreProvider>
    );
    console.log("[Main] Render dispatched.");
}
