/**
 * BaseRegistry: Motor genérico para gerenciamento de contratos.
 */
export class BaseRegistry {
  constructor(name, initialData = {}) {
    this.name = name;
    this.data = new Map(Object.entries(initialData));
  }

  get(key, fallback = null) {
    if (!key) return fallback;
    const normKey = String(key).toLowerCase().replace(/ /g, "_");
    if (this.data.has(normKey)) return this.data.get(normKey);
    if (this.data.has(key)) return this.data.get(key);
    return fallback || { title: key, tech: key, simple: null };
  }

  getAll() {
    return Object.fromEntries(this.data);
  }
}
