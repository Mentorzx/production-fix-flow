/**
 * Provide ExportService module functionality for the HPO dashboard.
 */

export const ExportService = {
  async export(format, data, filenameBase = "hpo_export") {
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, "-");
    const filename = `${filenameBase}_${timestamp}`;
    try {
      const response = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ format, filename, data }),
      });
      if (!response.ok) throw new Error(`Export HTTP error: ${response.status}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      const extMap = { toon: "txt", parquet: "parquet", json: "json", csv: "csv" };
      a.href = url;
      a.download = `${filename}.${extMap[format] || format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      return { success: true };
    } catch (error) {
      console.error("[ExportService] Falhou:", error);
      throw error;
    }
  },
};
