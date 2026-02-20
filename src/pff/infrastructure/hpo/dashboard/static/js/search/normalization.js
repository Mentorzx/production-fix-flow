/**
 * Search normalization utilities for command palette.
 */

const COMBINING_MARKS = /[\u0300-\u036f]/g;
const NON_WORD_SPLIT = /[^\p{L}\p{N}]+/gu;

/**
 * Normalize text without breaking Unicode letters.
 */
export const normalizeText = (value) => {
  const raw = String(value ?? "");
  return raw.normalize("NFD").replace(COMBINING_MARKS, "").toLowerCase().trim();
};

/**
 * Tokenize normalized text for inverted index lookup.
 */
export const tokenizeText = (value) => {
  const normalized = normalizeText(value);
  if (!normalized) return [];
  return normalized.split(NON_WORD_SPLIT).filter(Boolean);
};

/**
 * Build searchable blob from item fields.
 */
export const buildSearchBlob = ({ title = "", aliases = [], tags = [], description = "" }) => {
  return [title, ...(aliases || []), ...(tags || []), description].join(" ").trim();
};
