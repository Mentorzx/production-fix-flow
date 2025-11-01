from datetime import datetime
from pathlib import Path

import joblib

from pff import settings
from pff.utils import FileManager, logger


class EnsembleRulesExtractor:

    def __init__(self):
        self.file_manager = FileManager()
        self.feature_names = []
        self.rules_cache = []

    def extract_xgboost_rules(
        self,
        model,
        feature_names: list[str],
        max_depth: int = 3,
        min_confidence: float = 0.1,
    ) -> list[dict]:
        try:
            if not hasattr(model, "get_booster"):
                logger.error("Modelo não é um XGBoost válido")
                return []
            booster = model.get_booster()
            tree_data = booster.get_dump(dump_format="json")
            logger.info(f"🌳 Analisando {len(tree_data)} árvores do XGBoost")
            
            total_rules_before_filter = 0
            total_leaves_found = 0
            rules = []
            rule_id = 0
            
            for tree_idx, tree_json in enumerate(tree_data):
                # Sprint 16.5: Use FileManager for faster parsing (msgspec)
                tree = FileManager.json_loads(tree_json)
                tree_rules = self._extract_rules_from_tree(
                    tree, feature_names, tree_idx, max_depth, min_confidence
                )
                
                # Debug on first tree
                if tree_idx == 0:
                    logger.debug(f"Tree 0: extracted {len(tree_rules)} rules (max_depth={max_depth}, min_conf={min_confidence})")
                
                for rule in tree_rules:
                    rule["rule_id"] = f"ensemble_rule_{rule_id}"
                    rule["source"] = "xgboost_meta_learner"
                    rule["tree_index"] = tree_idx
                    rules.append(rule)
                    rule_id += 1
                    
            logger.info(f"✅ {len(rules)} regras extraídas do XGBoost")
            if len(rules) == 0:
                logger.warning(f"⚠️ Nenhuma regra extraída! Verifique max_depth={max_depth} e min_confidence={min_confidence}")
            return rules
        except Exception as e:
            logger.error(f"❌ Erro ao extrair regras do XGBoost: {e}")
            return []

    def _normalize_tree_node(self, node: dict) -> dict:
        """Normalize XGBoost tree node to handle multiple JSON formats.
        
        XGBoost tree format varies by version and parameters:
        - split: "split" vs "split_feature" vs "feature" vs "Split"
        - threshold: "split_condition" vs "threshold" vs "split_value" vs "Threshold"
        - children: "yes"/"no" vs "children" array vs "left"/"right"
        """
        if not isinstance(node, dict):
            return {}
        
        normalized = {}
        
        # Copy leaf value if present
        if "leaf" in node:
            normalized["leaf"] = node["leaf"]
            return normalized
        
        # Detect and normalize split field
        for field in ["split", "split_feature", "feature", "Split"]:
            if field in node:
                normalized["split"] = node[field]
                break
        
        # Detect and normalize threshold field
        for field in ["split_condition", "threshold", "split_value", "Threshold"]:
            if field in node:
                normalized["split_condition"] = node[field]
                break
        
        # Detect and normalize children fields
        if "yes" in node and "no" in node:
            normalized["yes"] = node["yes"]
            normalized["no"] = node["no"]
        elif "left" in node and "right" in node:
            normalized["yes"] = node["left"]
            normalized["no"] = node["right"]
        elif "children" in node and isinstance(node["children"], list) and len(node["children"]) == 2:
            normalized["yes"] = node["children"][0]
            normalized["no"] = node["children"][1]
        
        return normalized

    def _extract_rules_from_tree(
        self,
        tree_node: dict,
        feature_names: list[str],
        tree_idx: int,
        max_depth: int,
        min_confidence: float,
        path: list | None = None,
        depth: int = 0,
    ) -> list[dict]:
        if path is None:
            path = []
        rules = []
        
        # Log tree structure on first call for debugging
        if depth == 0 and tree_idx == 0:
            logger.debug(f"Tree node keys: {list(tree_node.keys())}")
        
        try:
            # Normalize node format
            node = self._normalize_tree_node(tree_node)
            
            if "leaf" in node:
                leaf_value = float(node["leaf"])
                confidence = abs(leaf_value)
                
                # Debug first tree
                if tree_idx == 0 and depth == 0:
                    logger.debug(f"Leaf node: value={leaf_value:.4f}, confidence={confidence:.4f}, path_len={len(path)}")
                
                if confidence >= min_confidence and len(path) > 0:
                    rule_text = self._path_to_prolog(path, leaf_value > 0)
                    rule = {
                        "prolog": rule_text,
                        "confidence": confidence,
                        "leaf_value": leaf_value,
                        "path_length": len(path),
                        "decision": "positive" if leaf_value > 0 else "negative",
                    }
                    rules.append(rule)
                elif tree_idx == 0:
                    if confidence < min_confidence:
                        logger.debug(f"  Skipped: confidence {confidence:.4f} < {min_confidence}")
                    if len(path) == 0:
                        logger.debug(f"  Skipped: empty path")
                return rules
            
            if depth < max_depth and "split" in node and "split_condition" in node:
                try:
                    # Handle XGBoost format: "f151" (string with 'f' prefix) or 151 (int)
                    split_value = node["split"]
                    if isinstance(split_value, str):
                        if split_value.startswith('f'):
                            feature_idx = int(split_value[1:])  # Remove 'f' prefix
                        else:
                            feature_idx = int(split_value)
                    else:
                        feature_idx = int(split_value)
                    
                    threshold = float(node["split_condition"])
                    
                    # Validate feature index
                    if not (0 <= feature_idx < len(feature_names)):
                        logger.warning(f"Invalid feature_idx {feature_idx} (max: {len(feature_names)-1})")
                        return rules
                    
                    feature_name = feature_names[feature_idx]
                    
                    if "yes" in node:
                        left_path = path + [(feature_name, "<", threshold)]
                        left_rules = self._extract_rules_from_tree(
                            node["yes"],
                            feature_names,
                            tree_idx,
                            max_depth,
                            min_confidence,
                            left_path,
                            depth + 1,
                        )
                        rules.extend(left_rules)
                    
                    if "no" in node:
                        right_path = path + [(feature_name, ">=", threshold)]
                        right_rules = self._extract_rules_from_tree(
                            node["no"],
                            feature_names,
                            tree_idx,
                            max_depth,
                            min_confidence,
                            right_path,
                            depth + 1,
                        )
                        rules.extend(right_rules)
                except (ValueError, TypeError, KeyError) as e:
                    logger.debug(f"Error extracting from tree node: {e}")
                    pass
        except Exception as e:
            logger.debug(f"Unexpected error in tree extraction: {e}")
            pass
        return rules

    def _path_to_prolog(self, path: list, is_positive: bool) -> str:
        head = "valid_data(X)" if is_positive else "invalid_data(X)"
        if not path:
            return f"{head} <= true"
        conditions = []
        for feature, operator, value in path:
            clean_feature = feature.replace("_", "").replace(" ", "")
            if operator == "<":
                condition = f"lessThan({clean_feature}(X), {value:.4f})"
            elif operator == ">=":
                condition = f"greaterEqual({clean_feature}(X), {value:.4f})"
            else:
                condition = f"{clean_feature}(X, {value})"
            conditions.append(condition)

        body = ", ".join(conditions)
        return f"{head} <= {body}"

    def load_manual_rules(self) -> list[dict]:
        manual_path = settings.PATTERNS_DIR / "manual_rules.json"
        if not manual_path.exists():
            logger.info("📂 Nenhum arquivo de regras manuais encontrado")
            return []
        try:
            data = self.file_manager.read(manual_path)
            manual_rules = data.get("rules", [])
            validated_rules = []
            for rule in manual_rules:
                if "prolog" in rule:
                    rule["source"] = "manual"
                    if "confidence" not in rule:
                        rule["confidence"] = 1.0
                    validated_rules.append(rule)
            logger.info(f"📂 {len(validated_rules)} regras manuais carregadas")
            return validated_rules
        except Exception as e:
            logger.error(f"❌ Erro ao carregar regras manuais: {e}")
            return []

    def extract_all_ensemble_rules(self, model_path: str | None = None) -> list[dict]:
        if model_path is None:
            model_path = str(
                settings.OUTPUTS_DIR / "ensemble" / "stacking_model_advanced.joblib"
            )
        try:
            logger.info("🚀 Iniciando extração completa de regras do ensemble")
            if not Path(model_path).exists():
                logger.error(f"❌ Modelo não encontrado: {model_path}")
                return self.load_manual_rules()
            ensemble_model = joblib.load(model_path)
            logger.info("✅ Modelo ensemble carregado")
            feature_names = self._get_feature_names(ensemble_model)
            meta_learner = ensemble_model.named_steps.get("meta_learner")
            if meta_learner is None:
                logger.error("❌ Meta-learner não encontrado no pipeline")
                return self.load_manual_rules()
            all_rules = []
            xgb_rules = self.extract_xgboost_rules(meta_learner, feature_names)
            all_rules.extend(xgb_rules)
            manual_rules = self.load_manual_rules()
            all_rules.extend(manual_rules)
            unique_rules = self._deduplicate_rules(all_rules)
            logger.info(f"🎉 Total de regras extraídas: {len(unique_rules)}")
            logger.info(f"   XGBoost: {len(xgb_rules)}")
            logger.info(f"   Manuais: {len(manual_rules)}")

            return unique_rules
        except Exception as e:
            logger.error(f"❌ Erro durante extração: {e}")
            return self.load_manual_rules()

    def _get_feature_names(self, ensemble_model) -> list[str]:
        try:
            feature_union = ensemble_model.named_steps.get("features")
            if feature_union:
                feature_names = ["hybrid_probability"]
                symbolic_transformer = None
                for name, transformer in feature_union.transformer_list:
                    if "symbolic" in name:
                        symbolic_transformer = transformer
                        break
                if symbolic_transformer and hasattr(symbolic_transformer, "rules_"):
                    num_rules = len(symbolic_transformer.rules_)
                    feature_names.extend([f"rule_{i}" for i in range(num_rules)])
                return feature_names
            return ["hybrid_probability"] + [f"rule_{i}" for i in range(100)]
        except Exception as e:
            logger.warning(f"⚠️ Erro ao obter feature names: {e}")
            return ["hybrid_probability"] + [f"rule_{i}" for i in range(100)]

    def _deduplicate_rules(self, rules: list[dict]) -> list[dict]:
        seen = set()
        unique_rules = []
        for rule in rules:
            prolog = rule.get("prolog", "").strip()
            if prolog and prolog not in seen:
                seen.add(prolog)
                unique_rules.append(rule)
        unique_rules.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return unique_rules

    def save_ensemble_rules(self, rules: list[dict]) -> Path:
        ensemble_data = {
            "rules": rules,
            "metadata": {
                "total_rules": len(rules),
                "sources": {
                    "xgboost": len(
                        [r for r in rules if r.get("source") == "xgboost_meta_learner"]
                    ),
                    "manual": len([r for r in rules if r.get("source") == "manual"]),
                },
                "extracted_at": datetime.now().isoformat(),
                "extractor_version": "2.0",
            },
        }
        output_path = settings.PATTERNS_DIR / "ensemble_rules.json"
        self.file_manager.save(ensemble_data, output_path)

        logger.success(f"✅ Regras do ensemble salvas: {output_path}")
        return output_path
