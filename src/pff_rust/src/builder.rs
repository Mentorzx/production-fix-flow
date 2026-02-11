//! KG Builder hot-path — Rust replacement for the DFS triple extraction in builder.py.
//!
//! The orchestration (I/O, Polars, splits) stays in Python.
//! This module accelerates `_convert_to_triples` (stack-based DFS over nested dicts/lists).
//!
//! Uses sonic-rs (SIMD-accelerated JSON parser, ~5x faster than serde_json for untyped parsing).

use pyo3::prelude::*;
use sonic_rs::{JsonContainerTrait, JsonValueTrait, Value};

const BAD_1: &str = "1970-01-01";
const BAD_2: &str = "9999-12-31";

/// Skip values that should not become triple objects.
fn is_skip_value(v: &str) -> bool {
    matches!(v, "{" | "[" | "{}" | "[]")
}

/// Check if a string contains sentinel date patterns.
#[inline]
fn has_bad_date(s: &str) -> bool {
    s.contains(BAD_1) || s.contains(BAD_2)
}

/// Clean text: collapse tabs to spaces and strip whitespace.
#[inline]
fn clean(text: &str) -> String {
    if text.contains('\t') {
        text.replace('\t', " ").trim().to_string()
    } else {
        text.trim().to_string()
    }
}

/// Extract an identifier (string or integer) from a JSON value.
#[inline]
fn extract_id(v: &Value) -> Option<String> {
    v.as_str()
        .map(|s| s.trim().to_string())
        .or_else(|| v.as_i64().map(|n| n.to_string()))
        .or_else(|| v.as_u64().map(|n| n.to_string()))
}

/// Format a numeric JSON value as a string.
#[inline]
fn number_to_string(v: &Value) -> String {
    v.as_i64()
        .map(|n| n.to_string())
        .or_else(|| v.as_u64().map(|n| n.to_string()))
        .or_else(|| v.as_f64().map(|n| n.to_string()))
        .unwrap_or_default()
}

/// Try to extract (s, p, o) triple from an object with those keys.
#[inline]
fn try_extract_spo(v: &Value) -> Option<(String, String, String)> {
    let sv = v.get("s");
    let pv = v.get("p");
    let ov = v.get("o");
    let s = sv.as_str()?;
    let p = pv.as_str()?;
    let o = ov.as_str()?;
    let sc = clean(s);
    let pc = clean(p);
    let oc = clean(o);
    if !sc.is_empty() && !pc.is_empty() && !oc.is_empty() {
        Some((sc, pc, oc))
    } else {
        None
    }
}

/// Convert a JSON object/string/list to (subject, triples) via stack-based DFS.
///
/// This is the hot-path function from builder.py._convert_to_triples, ported to Rust
/// for maximum throughput on large KG ingestion jobs.
///
/// Args:
///     json_str: JSON string representing the object to convert.
///     subject: Default subject identifier for this entity.
///
/// Returns:
///     List of (subject, predicate, object) string triples.
#[pyfunction]
pub fn convert_to_triples(json_str: &str, subject: &str) -> Vec<(String, String, String)> {
    let mut triples: Vec<(String, String, String)> = Vec::new();

    let parsed: Value = match sonic_rs::from_str(json_str) {
        Ok(v) => v,
        Err(_) => {
            parse_text_lines(json_str, subject, &mut triples);
            return triples;
        }
    };

    if let Some(arr) = parsed.as_array() {
        for item in arr.iter() {
            if let Some((sc, pc, oc)) = try_extract_spo(item)
                && !has_bad_date(&sc)
                && !has_bad_date(&pc)
                && !has_bad_date(&oc)
            {
                triples.push((sc, pc, oc));
            }
        }
    } else if parsed.is_object() {
        if let Some((sc, pc, oc)) = try_extract_spo(&parsed) {
            triples.push((sc, pc, oc));
            return triples;
        }

        let entity_id = parsed
            .get("id")
            .or_else(|| parsed.get("externalId"))
            .and_then(extract_id)
            .unwrap_or_else(|| subject.to_string());

        let obj = parsed.as_object().unwrap();
        let mut stack: Vec<(String, String, Value)> = Vec::new();
        for (k, v) in obj.iter() {
            if !k.starts_with('_') {
                stack.push((entity_id.clone(), k.to_string(), v.clone()));
            }
        }

        while let Some((subj, pred, val)) = stack.pop() {
            if val.is_null() {
                continue;
            }

            if let Some(s) = val.as_str() {
                let val_str = s.trim();
                if !val_str.is_empty() && !is_skip_value(val_str) {
                    let pred_clean = pred.trim().to_string();
                    if !has_bad_date(&subj) && !has_bad_date(&pred_clean) && !has_bad_date(val_str)
                    {
                        triples.push((subj, pred_clean, val_str.to_string()));
                    }
                }
            } else if let Some(b) = val.as_bool() {
                let pred_clean = pred.trim().to_string();
                if !has_bad_date(&subj) && !has_bad_date(&pred_clean) {
                    triples.push((subj, pred_clean, b.to_string()));
                }
            } else if val.is_number() {
                let pred_clean = pred.trim().to_string();
                if !has_bad_date(&subj) && !has_bad_date(&pred_clean) {
                    triples.push((subj, pred_clean, number_to_string(&val)));
                }
            } else if let Some(inner_obj) = val.as_object() {
                for (k, v) in inner_obj.iter() {
                    if !k.starts_with('_') {
                        stack.push((subj.clone(), format!("{pred}.{k}"), v.clone()));
                    }
                }
            } else if let Some(arr) = val.as_array() {
                for (idx, item) in arr.iter().enumerate() {
                    if item.is_object() {
                        let item_id = item
                            .get("id")
                            .or_else(|| item.get("externalId"))
                            .and_then(extract_id)
                            .unwrap_or_else(|| format!("{subj}_{pred}_{idx}"));

                        let pred_clean = pred.trim().to_string();
                        if !item_id.is_empty()
                            && !pred_clean.is_empty()
                            && !has_bad_date(&subj)
                            && !has_bad_date(&pred_clean)
                            && !has_bad_date(&item_id)
                        {
                            triples.push((subj.clone(), pred_clean, item_id.clone()));
                        }

                        let inner_obj = item.as_object().unwrap();
                        for (k, v) in inner_obj.iter() {
                            if !k.starts_with('_') {
                                stack.push((item_id.clone(), k.to_string(), v.clone()));
                            }
                        }
                    } else {
                        stack.push((subj.clone(), pred.clone(), item.clone()));
                    }
                }
            }
        }
    } else if let Some(s) = parsed.as_str() {
        parse_text_lines(s, subject, &mut triples);
    }

    triples
}

/// Parse text lines as tab-separated triples or key: value pairs.
fn parse_text_lines(text: &str, subject: &str, triples: &mut Vec<(String, String, String)>) {
    static SKIP_LINES: &[&str] = &["{", "}", "[", "]", "},", "],", "{}", "[]"];

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || SKIP_LINES.contains(&trimmed) {
            continue;
        }

        let parts: Vec<&str> = trimmed.splitn(3, '\t').collect();
        if parts.len() == 3 {
            let s = clean(parts[0]);
            let p = clean(parts[1]);
            let o = clean(parts[2]);
            if !s.is_empty() && !p.is_empty() && !o.is_empty() {
                triples.push((s, p, o));
            }
            continue;
        }

        if trimmed.starts_with('{')
            && trimmed.ends_with('}')
            && let Ok(parsed) = sonic_rs::from_str::<Value>(trimmed)
            && let Some((sc, pc, oc)) = try_extract_spo(&parsed)
        {
            triples.push((sc, pc, oc));
            continue;
        }

        if let Some(colon_pos) = trimmed.find(':') {
            let key = trimmed[..colon_pos]
                .trim()
                .trim_matches(|c| c == '"' || c == '\'');
            let val = trimmed[colon_pos + 1..]
                .trim()
                .trim_end_matches(',')
                .trim()
                .trim_matches(|c| c == '"' || c == '\'');

            if !key.is_empty()
                && !val.is_empty()
                && !is_skip_value(val)
                && !has_bad_date(subject)
                && !has_bad_date(key)
                && !has_bad_date(val)
            {
                triples.push((subject.to_string(), key.to_string(), val.to_string()));
            }
        }
    }
}
