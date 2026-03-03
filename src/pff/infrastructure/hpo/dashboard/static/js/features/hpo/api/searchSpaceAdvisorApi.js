/**
 * API adapter for Search Space Advisor feature.
 * Keeps HTTP concerns outside UI components.
 */

const JSON_HEADERS = { "Content-Type": "application/json" };

async function parseJsonResponse(response, fallbackMessage) {
  let data = null;
  try {
    data = await response.json();
  } catch {
    throw new Error(fallbackMessage);
  }

  if (!response.ok) {
    throw new Error(data?.detail || data?.error || fallbackMessage);
  }

  return data;
}

export async function previewSearchSpacePatch(recommendations) {
  const response = await fetch("/api/hpo/search-space-advice/patch", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ recommendations }),
  });

  return parseJsonResponse(response, "Falha ao gerar patch");
}

export async function refreshSearchSpaceAdvice() {
  const response = await fetch("/api/hpo/search-space-advice?refresh=1", {
    method: "GET",
    cache: "no-store",
  });

  return parseJsonResponse(response, "Falha ao recalcular recomendações.");
}

export async function applySearchSpaceRecommendations(recommendations) {
  const response = await fetch("/api/hpo/search-space-advice/apply", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ recommendations }),
  });

  return parseJsonResponse(response, "Falha ao aplicar ajustes");
}

export async function ignoreSearchSpaceRecommendation(paramName) {
  const response = await fetch("/api/hpo/search-space-advice/ignore", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ param_names: [paramName] }),
  });

  return parseJsonResponse(response, "Falha ao ignorar ajuste");
}
