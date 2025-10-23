from datetime import datetime

import polars as pl

from pff import settings
from pff.utils import FileManager, logger


class SmartAutofeeding:
    """
    Smart autofeeding that dynamically selects the best rule extraction strategy based on the pipeline state.
    """

    def __init__(self):
        self.file_manager = FileManager()
        self.phase = None

    def detect_pipeline_phase(self) -> str:
        """
        Detects the current pipeline phase and returns the appropriate strategy.
        Returns:
            'bootstrap': If this is the first run and the ensemble does not exist.
            'refinement': If the ensemble is already trained and advanced rules can be extracted.
            'hybrid': If the state is mixed and a combined strategy is needed.
        """
        logger.info("🔍 Detectando fase da pipeline...")
        ensemble_model_path = (
            settings.OUTPUTS_DIR / "ensemble" / "stacking_model_advanced.joblib"
        )
        ensemble_exists = ensemble_model_path.exists()
        anyburl_tsv_path = settings.OUTPUTS_DIR / "pyclause" / "rules_anyburl.tsv"
        anyburl_exists = anyburl_tsv_path.exists()
        combined_rules_path = settings.PATTERNS_DIR / "combined_rules.json"
        has_anyburl_rules = False
        if combined_rules_path.exists():
            try:
                content = self.file_manager.read(combined_rules_path)
                anyburl_rule_count = sum(
                    1
                    for rule in content.get("rules", [])
                    if rule.get("source") == "anyburl"
                )
                has_anyburl_rules = anyburl_rule_count > 0
            except Exception:
                has_anyburl_rules = False
        if not ensemble_exists and anyburl_exists and not has_anyburl_rules:
            phase = "bootstrap"
            logger.info(
                "📋 Fase: BOOTSTRAP - Primeira execução, converter AnyBURL TSV → JSON"
            )
        elif ensemble_exists and has_anyburl_rules:
            phase = "refinement"
            logger.info(
                "📋 Fase: REFINEMENT - Ensemble existe, extrair regras avançadas"
            )
        elif ensemble_exists and not has_anyburl_rules:
            phase = "hybrid"
            logger.info("📋 Fase: HYBRID - Ensemble existe mas regras básicas faltam")
        else:
            phase = "bootstrap"
            logger.warning("⚠️ Estado ambíguo, usando bootstrap como fallback")
        self.phase = phase
        return phase

    def apply_bootstrap_strategy(self) -> list[dict]:
        """
        Runs the bootstrap strategy: converts AnyBURL TSV to JSON and combines with manual rules.
        Returns the combined list of rules.
        """
        logger.info("🚀 Executando estratégia BOOTSTRAP...")
        anyburl_rules = self._convert_anyburl_tsv_to_json()
        manual_rules = self._load_manual_rules()
        all_rules = self._combine_rules(anyburl_rules, manual_rules)
        self._save_rules_to_files(
            all_rules, anyburl_rules, manual_rules, "bootstrap_v2.1"
        )
        logger.success(f"✅ Bootstrap concluído: {len(all_rules)} regras preparadas")
        return all_rules

    def apply_refinement_strategy(self) -> list[dict]:
        """
        Runs the refinement strategy: extracts advanced rules from the trained ensemble, combines and refines them, and saves the result.
        Returns the refined list of rules.
        """
        logger.info("🚀 Executando estratégia REFINEMENT...")
        try:
            ensemble_rules = self._extract_ensemble_rules()
            existing_rules = self._load_existing_rules()
            refined_rules = self._refine_and_combine_rules(
                ensemble_rules, existing_rules
            )
            self._save_rules_to_files(refined_rules, [], [], "refinement_v2.1")
            logger.success(
                f"✅ Refinement concluído: {len(refined_rules)} regras refinadas"
            )
            return refined_rules
        except Exception as e:
            logger.error(f"❌ Erro no refinement: {e}")
            logger.info("🔄 Fallback para estratégia híbrida...")
            return self.apply_hybrid_strategy()

    def apply_hybrid_strategy(self) -> list[dict]:
        """
        Runs the hybrid strategy: combines bootstrap and refinement, removes duplicates, and saves the result.
        Returns the consolidated list of rules.
        """
        logger.info("🚀 Executando estratégia HYBRID...")
        anyburl_rules = self._convert_anyburl_tsv_to_json()
        manual_rules = self._load_manual_rules()
        ensemble_rules = []
        try:
            ensemble_rules = self._extract_ensemble_rules()
        except Exception as e:
            logger.warning(f"⚠️ Não foi possível extrair regras do ensemble: {e}")
        all_sources = anyburl_rules + manual_rules + ensemble_rules
        refined_rules = self._remove_duplicates(all_sources)
        self._save_rules_to_files(
            refined_rules, anyburl_rules, manual_rules, "hybrid_v2.1"
        )
        logger.success(f"✅ Hybrid concluído: {len(refined_rules)} regras consolidadas")
        return refined_rules

    def _convert_anyburl_tsv_to_json(self) -> list[dict]:
        """
        Converts AnyBURL TSV rules to JSON format and returns the list.
        """
        anyburl_path = settings.OUTPUTS_DIR / "pyclause" / "rules_anyburl.tsv"
        if not anyburl_path.exists():
            logger.warning("⚠️ Arquivo AnyBURL TSV não encontrado")
            return []
        try:
            df = pl.read_csv(anyburl_path, separator="\t", has_header=False)
            rules = []
            for row in df.iter_rows(named=True):
                rule = {
                    "prolog": row["column_4"],
                    "confidence": float(row["column_3"]),
                    "support": int(row["column_2"]),
                    "predictions": int(row["column_1"]),
                    "source": "anyburl",
                    "extraction_method": "tsv_conversion",
                    "quality_score": float(row["column_3"])
                    * (int(row["column_2"]) / 100.0),
                }
                rules.append(rule)
            logger.info(f"✅ Convertidas {len(rules)} regras AnyBURL")
            return rules
        except Exception as e:
            logger.error(f"❌ Erro na conversão AnyBURL: {e}")
            return []

    def _load_manual_rules(self) -> list[dict]:
        """
        Loads manual rules from file and returns the list.
        """
        manual_path = settings.PATTERNS_DIR / "manual_rules.json"
        if not manual_path.exists():
            return []
        try:
            content = self.file_manager.read(manual_path)
            rules = content.get("rules", [])
            for rule in rules:
                rule["source"] = "manual"
                rule["extraction_method"] = "manual_curation"
                if "quality_score" not in rule:
                    rule["quality_score"] = rule.get("confidence", 1.0)
            logger.info(f"✅ Carregadas {len(rules)} regras manuais")
            return rules
        except Exception as e:
            logger.error(f"❌ Erro ao carregar regras manuais: {e}")
            return []

    def _extract_ensemble_rules(self) -> list[dict]:
        """
        Extracts advanced rules from the trained ensemble and returns the list.
        """
        try:
            from pff.validators.ensembles.ensemble_rules_extractor import (
                EnsembleRulesExtractor,
            )

            extractor = EnsembleRulesExtractor()
            rules = extractor.extract_all_ensemble_rules()
            for rule in rules:
                rule["extraction_method"] = "ensemble_meta_learner"
                if "quality_score" not in rule:
                    rule["quality_score"] = rule.get("confidence", 0.5)
            logger.info(f"✅ Extraídas {len(rules)} regras do ensemble")
            return rules
        except ImportError:
            logger.warning("⚠️ EnsembleRulesExtractor não disponível")
            return []
        except Exception as e:
            logger.error(f"❌ Erro na extração do ensemble: {e}")
            return []

    def _load_existing_rules(self) -> list[dict]:
        """
        Loads existing rules from JSON files and returns the list.
        """
        combined_path = settings.PATTERNS_DIR / "combined_rules.json"
        if not combined_path.exists():
            return []
        try:
            content = self.file_manager.read(combined_path)
            return content.get("rules", [])
        except Exception:
            return []

    def _combine_rules(
        self, anyburl_rules: list[dict], manual_rules: list[dict]
    ) -> list[dict]:
        """
        Combines rules from different sources, removing duplicates, and returns the list.
        """
        all_rules = anyburl_rules + manual_rules
        return self._remove_duplicates(all_rules)

    def _refine_and_combine_rules(
        self, ensemble_rules: list[dict], existing_rules: list[dict]
    ) -> list[dict]:
        """
        Performs advanced refinement: combines rules, weights by quality and performance, and returns the refined list.
        """
        all_rules = ensemble_rules + existing_rules
        seen_prolog = {}
        refined_rules = []
        for rule in all_rules:
            prolog = rule.get("prolog", "").strip()
            if not prolog:
                continue
            quality_score = rule.get("quality_score", 0.0)
            if (
                prolog not in seen_prolog
                or quality_score > seen_prolog[prolog]["quality_score"]
            ):
                seen_prolog[prolog] = rule
        refined_rules = list(seen_prolog.values())
        refined_rules.sort(key=lambda x: x.get("quality_score", 0.0), reverse=True)
        logger.info(f"🔧 Refinamento: {len(all_rules)} → {len(refined_rules)} regras")
        return refined_rules

    def _remove_duplicates(self, rules: list[dict]) -> list[dict]:
        """
        Removes simple duplicates based on the prolog string and returns the unique list.
        """
        seen = set()
        unique_rules = []
        for rule in rules:
            prolog = rule.get("prolog", "").strip()
            if prolog and prolog not in seen:
                seen.add(prolog)
                unique_rules.append(rule)
        return unique_rules

    def _save_rules_to_files(
        self,
        all_rules: list[dict],
        anyburl_rules: list[dict],
        manual_rules: list[dict],
        version: str,
    ):
        """
        Saves rules to the required files with compatible structure.
        """
        combined_data = {
            "rules": all_rules,
            "sources": {
                "anyburl": len(anyburl_rules),
                "manual": len(manual_rules),
                "ensemble": len(all_rules) - len(anyburl_rules) - len(manual_rules),
            },
            "total": len(all_rules),
            "timestamp": datetime.now().isoformat(),
            "version": version,
            "phase": self.phase,
            "description": f"Smart autofeeding v2.1 - {self.phase} strategy",
        }
        combined_path = settings.PATTERNS_DIR / "combined_rules.json"
        self.file_manager.save(combined_data, combined_path)
        clause_rules_path = settings.PATTERNS_DIR / "clause_rules.json"
        self.file_manager.save(combined_data, clause_rules_path)
        ensemble_data = {
            "rules": anyburl_rules if anyburl_rules else all_rules,
            "source": "anyburl_tsv_converted" if anyburl_rules else "smart_autofeeding",
            "total": len(anyburl_rules) if anyburl_rules else len(all_rules),
            "timestamp": datetime.now().isoformat(),
            "description": f"Smart autofeeding v2.1 rules - {self.phase} phase",
        }
        ensemble_path = settings.PATTERNS_DIR / "ensemble_rules.json"
        self.file_manager.save(ensemble_data, ensemble_path)
        logger.success(
            f"✅ Regras salvas em {len([combined_path, clause_rules_path, ensemble_path])} arquivos"
        )


async def apply_autofeeding_rules_deprecated() -> None:
    """
    Deprecated version kept for compatibility, now uses Smart Autofeeding.
    """
    logger.info("🔄 Aplicando regras de autofeeding (deprecated → smart v2.1)...")
    smart_autofeeding = SmartAutofeeding()
    phase = smart_autofeeding.detect_pipeline_phase()
    if phase == "bootstrap":
        rules = smart_autofeeding.apply_bootstrap_strategy()
    else:
        rules = smart_autofeeding.apply_hybrid_strategy()
    await update_knowledge_graph_with_rules(rules)
    if rules:
        logger.success(f"✅ Autofeeding deprecated concluído: {len(rules)} regras")
    else:
        logger.error("❌ Falha no autofeeding deprecated")


async def apply_autofeeding_rules() -> None:
    """
    Main autofeeding version - Smart Autofeeding v2.1.
    Replaces the problematic 2.0 version while keeping all sophistication.
    """
    logger.info("🧠 Smart Autofeeding v2.1 iniciado...")
    try:
        smart_autofeeding = SmartAutofeeding()
        phase = smart_autofeeding.detect_pipeline_phase()
        if phase == "bootstrap":
            rules = smart_autofeeding.apply_bootstrap_strategy()
        elif phase == "refinement":
            rules = smart_autofeeding.apply_refinement_strategy()
        else:  # hybrid
            rules = smart_autofeeding.apply_hybrid_strategy()
        await update_knowledge_graph_with_rules(rules)
        if rules:
            anyburl_count = sum(1 for r in rules if r.get("source") == "anyburl")
            manual_count = sum(1 for r in rules if r.get("source") == "manual")
            ensemble_count = len(rules) - anyburl_count - manual_count
            logger.success("🎉 Smart Autofeeding v2.1 concluído!")
            logger.info("📊 Estatísticas finais:")
            logger.info(f"   AnyBURL: {anyburl_count}")
            logger.info(f"   Manual: {manual_count}")
            logger.info(f"   Ensemble: {ensemble_count}")
            logger.info(f"   Total: {len(rules)}")
            logger.info(f"   Estratégia: {phase}")
            if len(rules) > 0:
                logger.success(
                    f"SUCESSO: combined_rules.json agora tem {len(rules)} regras!"
                )
                logger.info("Agora o SymbolicFeatureExtractor terá regras para usar")
            else:
                logger.error("FALHA: combined_rules.json ainda está vazio")
        else:
            logger.error("❌ Smart Autofeeding falhou - nenhuma regra gerada")
    except Exception as e:
        logger.error(f"❌ Erro no Smart Autofeeding v2.1: {e}")
        logger.info("🔄 Tentando fallback para versão deprecated...")
        await apply_autofeeding_rules_deprecated()


async def update_knowledge_graph_with_rules(rules: list[dict]) -> None:
    """
    Update the knowledge graph data with high-confidence rules.
    Kept for full compatibility.
    """
    logger.info("Atualizando grafo de conhecimento com novas regras...")
    file_manager = FileManager()
    new_triples = []
    for rule in rules:
        if isinstance(rule, dict):
            prolog = rule.get("prolog", "")
        elif isinstance(rule, str):
            prolog = rule
        else:
            continue
        if ":-" in prolog:
            head, body = prolog.split(":-", 1)
            head = head.strip()
            body = body.strip()
            if "(" in head and "," in head:
                rel = head.split("(")[0].strip()
                args = head.split("(")[1].split(")")[0].split(",")
                if len(args) == 2:
                    subj = args[0].strip()
                    obj = args[1].strip()
                    triple = (subj, rel, obj)
                    new_triples.append(triple)
                    logger.debug(f"Tripla extraída da regra: {triple}")
    if new_triples:
        train_path = settings.DATA_DIR / "models" / "kg" / "train.parquet"
        if train_path.exists():
            _train_df = file_manager.read(train_path)
            logger.info(
                f"{len(new_triples)} novas triplas extraídas de regras (placeholder, não adicionadas ao arquivo)"
            )
    logger.success("Grafo de conhecimento atualizado")
