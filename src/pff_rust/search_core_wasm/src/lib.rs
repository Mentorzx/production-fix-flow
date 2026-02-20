use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};

use boomphf::Mphf;
use nucleo_matcher::{Config, Matcher, Utf32Str};
use serde::{Deserialize, Serialize};
use unicode_normalization::{char::is_combining_mark, UnicodeNormalization};
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchItemInput {
    id: String,
    dom_id: String,
    title: String,
    description: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    section_path: String,
    #[serde(default)]
    section_key: String,
    #[serde(default)]
    tab_id: String,
    #[serde(default)]
    view_mode: String,
}

#[derive(Debug, Clone)]
struct IndexedItem {
    raw: SearchItemInput,
    title_norm: String,
    description_norm: String,
    aliases_norm: Vec<String>,
    tags_norm: Vec<String>,
    search_blob_norm: String,
    token_set: HashSet<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SearchSuggestion {
    id: String,
    dom_id: String,
    title: String,
    snippet: String,
    score: f64,
    tab_id: String,
    view_mode: String,
    section_path: String,
    section_key: String,
    reason: String,
}

#[derive(Debug)]
struct SearchEngine {
    items: Vec<IndexedItem>,
    all_bits: u64,
    exact_keys: Vec<String>,
    exact_bits: Vec<u64>,
    exact_set: HashSet<String>,
    exact_mphf: Option<Mphf<String>>,
    token_index: HashMap<String, u64>,
    matcher: Matcher,
    haystack_buf: Vec<char>,
    needle_buf: Vec<char>,
}

#[wasm_bindgen]
pub struct SearchEngineHandle {
    engine: RefCell<SearchEngine>,
}

#[wasm_bindgen]
pub fn init_catalog(items: JsValue) -> Result<SearchEngineHandle, JsValue> {
    let parsed: Vec<SearchItemInput> =
        serde_wasm_bindgen::from_value(items).map_err(js_error_from_serde)?;
    let engine = SearchEngine::new(parsed).map_err(js_error)?;
    Ok(SearchEngineHandle {
        engine: RefCell::new(engine),
    })
}

#[wasm_bindgen]
pub fn search(handle: &SearchEngineHandle, query: String, k: u32) -> Result<JsValue, JsValue> {
    let take = if k == 0 { 10 } else { k as usize };
    let mut engine = handle.engine.borrow_mut();
    let suggestions = engine.search(&query, take);
    serde_wasm_bindgen::to_value(&suggestions).map_err(js_error_from_serde)
}

impl SearchEngine {
    fn new(items: Vec<SearchItemInput>) -> Result<Self, String> {
        if items.is_empty() {
            return Err("catalog cannot be empty".to_string());
        }
        if items.len() > 64 {
            return Err(format!(
                "catalog size {} exceeds 64-item bitset limit",
                items.len()
            ));
        }

        let indexed_items: Vec<IndexedItem> = items
            .into_iter()
            .map(|item| {
                let title_norm = normalize_text(&item.title);
                let description_norm = normalize_text(&item.description);
                let aliases_norm: Vec<String> = item.aliases.iter().map(|v| normalize_text(v)).collect();
                let tags_norm: Vec<String> = item.tags.iter().map(|v| normalize_text(v)).collect();
                let search_blob_norm = [
                    title_norm.as_str(),
                    aliases_norm.join(" ").as_str(),
                    tags_norm.join(" ").as_str(),
                    description_norm.as_str(),
                ]
                .join(" ");

                let token_set: HashSet<String> = tokenize(&search_blob_norm).into_iter().collect();

                IndexedItem {
                    raw: item,
                    title_norm,
                    description_norm,
                    aliases_norm,
                    tags_norm,
                    search_blob_norm,
                    token_set,
                }
            })
            .collect();

        let mut exact_multimap: BTreeMap<String, u64> = BTreeMap::new();
        let mut token_index: HashMap<String, u64> = HashMap::new();

        for (idx, item) in indexed_items.iter().enumerate() {
            let bit = bit_for(idx)?;
            exact_multimap
                .entry(item.title_norm.clone())
                .and_modify(|mask| *mask |= bit)
                .or_insert(bit);
            for alias in &item.aliases_norm {
                if alias.is_empty() {
                    continue;
                }
                exact_multimap
                    .entry(alias.clone())
                    .and_modify(|mask| *mask |= bit)
                    .or_insert(bit);
            }

            for token in &item.token_set {
                token_index
                    .entry(token.clone())
                    .and_modify(|mask| *mask |= bit)
                    .or_insert(bit);
            }
        }

        let ordered_keys: Vec<String> = exact_multimap.keys().cloned().collect();
        let exact_set: HashSet<String> = ordered_keys.iter().cloned().collect();
        let (exact_keys, exact_bits, exact_mphf) = if ordered_keys.is_empty() {
            (Vec::new(), Vec::new(), None)
        } else {
            let mphf = Mphf::new(1.7, &ordered_keys);
            let mut mphf_keys = vec![String::new(); ordered_keys.len()];
            let mut mphf_bits = vec![0u64; ordered_keys.len()];
            for key in ordered_keys {
                let idx = mphf.hash(&key) as usize;
                mphf_keys[idx] = key.clone();
                mphf_bits[idx] = *exact_multimap.get(&key).unwrap_or(&0);
            }
            (mphf_keys, mphf_bits, Some(mphf))
        };

        let all_bits = if indexed_items.is_empty() {
            0
        } else {
            ((1u128 << indexed_items.len()) - 1) as u64
        };

        Ok(Self {
            items: indexed_items,
            all_bits,
            exact_keys,
            exact_bits,
            exact_set,
            exact_mphf,
            token_index,
            matcher: Matcher::new(Config::DEFAULT),
            haystack_buf: Vec::new(),
            needle_buf: Vec::new(),
        })
    }

    fn search(&mut self, query: &str, k: usize) -> Vec<SearchSuggestion> {
        let query_norm = normalize_text(query).trim().to_string();
        if query_norm.is_empty() {
            return Vec::new();
        }

        if let Some(exact_bits) = self.fast_path_match_bits(&query_norm) {
            let mut exact_ranked = self.rank_bits(exact_bits, &query_norm, "exact_mphf");
            exact_ranked.truncate(k);
            return exact_ranked;
        }

        let query_tokens = tokenize(&query_norm);
        let candidate_bits = self.pre_filter_candidates(&query_tokens);
        let mut ranked = self.rank_bits(candidate_bits, &query_norm, "token_fuzzy");
        ranked.truncate(k);
        ranked
    }

    fn fast_path_match_bits(&self, query_norm: &str) -> Option<u64> {
        let mphf = self.exact_mphf.as_ref()?;
        if !self.exact_set.contains(query_norm) {
            return None;
        }
        let idx = mphf.hash(&query_norm.to_string()) as usize;
        if idx >= self.exact_keys.len() {
            return None;
        }
        if self.exact_keys[idx] != query_norm {
            return None;
        }
        let bits = self.exact_bits[idx];
        if bits == 0 {
            return None;
        }
        Some(bits)
    }

    fn pre_filter_candidates(&self, query_tokens: &[String]) -> u64 {
        if query_tokens.is_empty() {
            return self.all_bits;
        }

        let mut and_bits: u64 = 0;
        let mut saw_any = false;
        for token in query_tokens {
            if let Some(bits) = self.token_index.get(token) {
                and_bits = if saw_any { and_bits & *bits } else { *bits };
                saw_any = true;
            }
        }

        if saw_any && and_bits != 0 {
            return and_bits;
        }

        let mut or_bits: u64 = 0;
        for token in query_tokens {
            if let Some(bits) = self.token_index.get(token) {
                or_bits |= *bits;
            }
        }

        if or_bits != 0 {
            return or_bits;
        }

        self.all_bits
    }

    fn rank_bits(&mut self, bits: u64, query_norm: &str, reason: &str) -> Vec<SearchSuggestion> {
        let mut ranked: Vec<(usize, f64)> = Vec::new();
        for idx in bits_to_indices(bits) {
            if let Some(score) = self.fuzzy_score_for_item(idx, query_norm) {
                ranked.push((idx, score));
            }
        }

        ranked.sort_by(|(left_idx, left_score), (right_idx, right_score)| {
            right_score
                .partial_cmp(left_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    self.items[*left_idx]
                        .raw
                        .title
                        .cmp(&self.items[*right_idx].raw.title)
                })
        });

        ranked
            .into_iter()
            .map(|(idx, score)| self.to_suggestion(idx, score, reason))
            .collect()
    }

    fn fuzzy_score_for_item(&mut self, idx: usize, query_norm: &str) -> Option<f64> {
        let item = self.items.get(idx)?;
        let needle = Utf32Str::new(query_norm, &mut self.needle_buf);

        let mut title_score = 0f64;
        let mut aliases_score = 0f64;
        let mut tags_score = 0f64;
        let mut description_score = 0f64;
        let mut blob_score = 0f64;

        if let Some(score) = fuzzy_score(
            &mut self.matcher,
            &item.title_norm,
            needle,
            &mut self.haystack_buf,
        ) {
            title_score = score;
        }

        for alias in &item.aliases_norm {
            if let Some(score) = fuzzy_score(&mut self.matcher, alias, needle, &mut self.haystack_buf) {
                aliases_score = aliases_score.max(score);
            }
        }

        for tag in &item.tags_norm {
            if let Some(score) = fuzzy_score(&mut self.matcher, tag, needle, &mut self.haystack_buf) {
                tags_score = tags_score.max(score);
            }
        }

        if let Some(score) = fuzzy_score(
            &mut self.matcher,
            &item.description_norm,
            needle,
            &mut self.haystack_buf,
        ) {
            description_score = score;
        }

        if let Some(score) = fuzzy_score(
            &mut self.matcher,
            &item.search_blob_norm,
            needle,
            &mut self.haystack_buf,
        ) {
            blob_score = score;
        }

        if title_score <= 0.0
            && aliases_score <= 0.0
            && tags_score <= 0.0
            && description_score <= 0.0
            && blob_score <= 0.0
        {
            return None;
        }

        let query_tokens = tokenize(query_norm);
        let matched_tokens = query_tokens
            .iter()
            .filter(|token| item.token_set.contains(token.as_str()))
            .count();
        let coverage = if query_tokens.is_empty() {
            0.0
        } else {
            matched_tokens as f64 / query_tokens.len() as f64
        };

        let weighted = title_score * 4.0
            + aliases_score * 3.0
            + tags_score * 2.0
            + description_score * 1.5
            + blob_score;

        Some(weighted + coverage * 120.0)
    }

    fn to_suggestion(&self, idx: usize, score: f64, reason: &str) -> SearchSuggestion {
        let item = &self.items[idx].raw;
        SearchSuggestion {
            id: item.id.clone(),
            dom_id: item.dom_id.clone(),
            title: item.title.clone(),
            snippet: snippet(&item.description),
            score,
            tab_id: item.tab_id.clone(),
            view_mode: item.view_mode.clone(),
            section_path: item.section_path.clone(),
            section_key: item.section_key.clone(),
            reason: reason.to_string(),
        }
    }
}

fn fuzzy_score(
    matcher: &mut Matcher,
    haystack: &str,
    needle: Utf32Str<'_>,
    haystack_buf: &mut Vec<char>,
) -> Option<f64> {
    if haystack.is_empty() {
        return None;
    }
    let hay = Utf32Str::new(haystack, haystack_buf);
    matcher.fuzzy_match(hay, needle).map(|score| score as f64)
}

fn normalize_text(input: &str) -> String {
    input
        .nfd()
        .filter(|c| !is_combining_mark(*c))
        .collect::<String>()
        .to_lowercase()
}

fn tokenize(input: &str) -> Vec<String> {
    input
        .split(|c: char| !c.is_alphanumeric())
        .filter_map(|token| {
            let trimmed = token.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        })
        .collect()
}

fn bits_to_indices(bits: u64) -> Vec<usize> {
    let mut result = Vec::new();
    let mut mask = bits;
    while mask != 0 {
        let tz = mask.trailing_zeros() as usize;
        result.push(tz);
        mask &= mask - 1;
    }
    result
}

fn bit_for(idx: usize) -> Result<u64, String> {
    if idx >= 64 {
        return Err(format!("index {idx} exceeds u64 bitset capacity"));
    }
    Ok(1u64 << idx)
}

fn snippet(description: &str) -> String {
    let trimmed = description.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut chars = trimmed.chars();
    let snippet: String = chars.by_ref().take(120).collect();
    if chars.next().is_some() {
        format!("{snippet}...")
    } else {
        snippet
    }
}

fn js_error(message: impl Into<String>) -> JsValue {
    JsValue::from_str(&message.into())
}

fn js_error_from_serde(error: serde_wasm_bindgen::Error) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn test_catalog() -> Vec<SearchItemInput> {
        vec![
            SearchItemInput {
                id: "vendas-regiao".to_string(),
                dom_id: "search-vendas-regiao".to_string(),
                title: "Vendas por Região".to_string(),
                description: "Gráfico que mostra vendas por região e tendências históricas."
                    .to_string(),
                aliases: vec!["vendas regiao".to_string(), "receita regional".to_string()],
                tags: vec!["vendas".to_string(), "regiao".to_string()],
                section_path: "Análise > Comercial".to_string(),
                section_key: "analise-comercial".to_string(),
                tab_id: "analysis".to_string(),
                view_mode: "study".to_string(),
            },
            SearchItemInput {
                id: "latencia".to_string(),
                dom_id: "search-latencia".to_string(),
                title: "Latência por Trial".to_string(),
                description: "Tabela com latência média e p95 por trial.".to_string(),
                aliases: vec!["tempo resposta".to_string()],
                tags: vec!["latencia".to_string(), "p95".to_string()],
                section_path: "Análise > Diagnóstico".to_string(),
                section_key: "analise-diagnostico".to_string(),
                tab_id: "analysis".to_string(),
                view_mode: "study".to_string(),
            },
        ]
    }

    #[test]
    fn exact_match_uses_mphf() {
        let mut engine = SearchEngine::new(test_catalog()).expect("engine should build");
        let results = engine.search("vendas por regiao", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "vendas-regiao");
        assert_eq!(results[0].reason, "exact_mphf");
    }

    #[test]
    fn token_index_filters_candidates() {
        let engine = SearchEngine::new(test_catalog()).expect("engine should build");
        let bits = engine.pre_filter_candidates(&["regiao".to_string()]);
        let indices = bits_to_indices(bits);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn fuzzy_ranks_close_query_on_top() {
        let mut engine = SearchEngine::new(test_catalog()).expect("engine should build");
        let results = engine.search("vend reg", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "vendas-regiao");
    }

    #[test]
    fn mphf_guard_prevents_false_positive_on_missing_key() {
        let engine = SearchEngine::new(test_catalog()).expect("engine should build");
        assert!(engine.fast_path_match_bits("nao-existe") .is_none());
    }

    #[test]
    fn search_latency_stays_under_five_ms_for_small_catalog() {
        let mut catalog = test_catalog();
        for idx in 0..30 {
            catalog.push(SearchItemInput {
                id: format!("extra-{idx}"),
                dom_id: format!("search-extra-{idx}"),
                title: format!("Painel Extra {idx}"),
                description: "Painel auxiliar para benchmark local da command palette.".to_string(),
                aliases: vec![format!("extra painel {idx}")],
                tags: vec!["extra".to_string(), "benchmark".to_string()],
                section_path: "Benchmark".to_string(),
                section_key: "bench".to_string(),
                tab_id: "analysis".to_string(),
                view_mode: "study".to_string(),
            });
        }

        let mut engine = SearchEngine::new(catalog).expect("engine should build");
        let runs = 800u32;
        let start = Instant::now();
        for _ in 0..runs {
            let _ = engine.search("vend reg", 10);
        }
        let avg_ms = (start.elapsed().as_secs_f64() * 1000.0) / f64::from(runs);
        assert!(
            avg_ms < 5.0,
            "average search latency is too high: {avg_ms:.4}ms"
        );
    }
}
