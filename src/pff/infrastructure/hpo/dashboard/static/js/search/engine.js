/**
 * Search engine bootstrap: prefers Rust/WASM core, falls back to local JS matcher.
 */

import { normalizeText } from "./normalization.js";

let wasmModulePromise = null;

const buildVersionQuery = () => {
  const buildId = typeof window !== "undefined" ? window.__PFF_BUILD_ID__ : "";
  return buildId ? `?${encodeURIComponent(String(buildId))}` : "";
};

const loadWasmModule = async () => {
  if (wasmModulePromise) return wasmModulePromise;
  const query = buildVersionQuery();
  wasmModulePromise = import(`/dist/search_core.js${query}`).then(async (mod) => {
    if (typeof mod.default === "function") {
      await mod.default(`/dist/search_core_bg.wasm${query}`);
    }
    return mod;
  });
  return wasmModulePromise;
};

const jsScore = (item, queryNorm) => {
  if (!queryNorm) return 0;
  const title = item.titleNorm || "";
  const aliases = item.aliasesNorm || [];
  const tags = item.tagsNorm || [];
  const description = item.descriptionNorm || "";
  const blob = item.searchBlobNorm || "";

  let score = 0;
  if (title === queryNorm) score += 500;
  if (title.includes(queryNorm)) score += 220;

  for (const alias of aliases) {
    if (alias === queryNorm) score += 420;
    if (alias.includes(queryNorm)) score += 180;
  }

  for (const tag of tags) {
    if (tag === queryNorm) score += 140;
    if (tag.includes(queryNorm)) score += 70;
  }

  if (description.includes(queryNorm)) score += 90;

  const queryTokens = queryNorm.split(/\s+/).filter(Boolean);
  let tokenHits = 0;
  for (const token of queryTokens) {
    if (blob.includes(token)) {
      tokenHits += 1;
      score += 24;
    }
  }

  if (queryTokens.length > 0) {
    score += (tokenHits / queryTokens.length) * 80;
  }

  return score;
};

const buildSnippet = (description) => {
  const text = String(description || "").trim();
  if (!text) return "";
  return text.length > 120 ? `${text.slice(0, 120)}...` : text;
};

const jsFallbackSearch = (catalog, query, k = 10) => {
  const queryNorm = normalizeText(query);
  if (!queryNorm) return [];

  return catalog
    .map((item) => ({ item, score: jsScore(item, queryNorm) }))
    .filter((entry) => entry.score > 0)
    .sort((left, right) => right.score - left.score || left.item.title.localeCompare(right.item.title))
    .slice(0, k)
    .map(({ item, score }) => ({
      id: item.id,
      domId: item.domId,
      title: item.title,
      snippet: buildSnippet(item.description),
      score,
      tabId: item.tabId,
      viewMode: item.viewMode,
      sectionPath: item.sectionPath,
      sectionKey: item.sectionKey || "",
      reason: "js_fallback",
    }));
};

/**
 * Initialize search engine context.
 */
export const initSearchEngine = async (catalog) => {
  const safeCatalog = Array.isArray(catalog) ? catalog.slice(0, 64) : [];
  if (safeCatalog.length === 0) {
    return { mode: "empty", catalog: [] };
  }

  try {
    const wasm = await loadWasmModule();
    const handle = wasm.init_catalog(safeCatalog);
    return {
      mode: "wasm",
      wasm,
      handle,
      catalog: safeCatalog,
    };
  } catch (error) {
    console.error("[SearchPalette] WASM search disabled, using JS fallback", error);
    return {
      mode: "js",
      catalog: safeCatalog,
      error: String(error?.message || error),
    };
  }
};

/**
 * Execute search query and return normalized suggestions.
 */
export const searchWithEngine = (engine, query, k = 10) => {
  if (!engine || !Array.isArray(engine.catalog)) return [];

  if (engine.mode === "wasm" && engine.wasm && engine.handle) {
    try {
      const results = engine.wasm.search(engine.handle, String(query || ""), k);
      return Array.isArray(results) ? results : [];
    } catch (error) {
      console.error("[SearchPalette] WASM query failed, degrading to JS fallback", error);
      return jsFallbackSearch(engine.catalog, query, k);
    }
  }

  return jsFallbackSearch(engine.catalog, query, k);
};
