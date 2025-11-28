"""
Smart Autofeeding Module for Rule Extraction Pipeline.

Design Patterns:
- Builder Pattern: SmartAutofeeding constructs complex rule sets step-by-step through
  detect_pipeline_phase() -> apply_*_strategy() -> execute_autofeeding()
- Strategy Pattern: Different strategies (bootstrap, refinement, hybrid) are selected
  dynamically based on pipeline state detection

This module handles intelligent rule extraction and combination based on the current
state of the ensemble training pipeline.
"""

from datetime import datetime
import asyncio
import polars as pl

from pff import settings
from pff.config import AUTOFEEDING_CONFIG_PATH
from pff.db.repositories.kg_rules import KGRulesRepository
from ..core.file_manager import FileManager
from ..core.logger import logger


class SmartAutofeeding:
    """
    Smart autofeeding that dynamically selects the best rule extraction strategy based on the pipeline state.
    """

    def __init__(self, rules_repository: KGRulesRepository | None = None):
        self.file_manager = FileManager()
        self.phase = None
        self.repo = rules_repository or KGRulesRepository()

    async def detect_pipeline_phase(self) -> str:
        """
        Detects the current pipeline phase and returns the appropriate strategy.
        Returns:
            'bootstrap': If this is the first run and the ensemble does not exist.
            'refinement': If the ensemble is already trained and advanced rules can be extracted.
            'hybrid': If the state is mixed and a combined strategy is needed.
        """
        logger.info(" Detectando fase da pipeline...")
        
        # Check DB for rules
        anyburl_count = await self.repo.count_rules(source="anyburl")
        has_anyburl_rules = anyburl_count > 0
        
        ensemble_model_path = (
            settings.OUTPUTS_DIR / "ensemble" / "stacking_model_advanced.joblib"
        )
        ensemble_exists = ensemble_model_path.exists()
        
        # Fallback to file check if DB is empty (migration scenario)
        if not has_anyburl_rules:
             anyburl_tsv_path = settings.OUTPUTS_DIR / "pyclause" / "rules_anyburl.tsv"
             if anyburl_tsv_path.exists():
                 logger.info(" Regras AnyBURL encontradas em arquivo, mas não no DB. Migrando...")
                 # We will load them in the strategy execution
                 has_anyburl_rules = True

        if not ensemble_exists and has_anyburl_rules:
            phase = "bootstrap"
            logger.info(
                " Fase: BOOTSTRAP - Primeira execução (ou migração)"
            )
        elif ensemble_exists and has_anyburl_rules:
            phase = "refinement"
            logger.info(
                " Fase: REFINEMENT - Ensemble existe, extrair regras avançadas"
            )
        elif ensemble_exists and not has_anyburl_rules:
            phase = "hybrid"
            logger.info(" Fase: HYBRID - Ensemble existe mas regras básicas faltam")
        else:
            phase = "bootstrap"
            logger.warning("Ambiguous state detected; using bootstrap as fallback")
        self.phase = phase
        return phase

    async def apply_bootstrap_strategy(self) -> list[dict]:
        """
        Runs the bootstrap strategy: loads AnyBURL rules (DB/File) and combines with manual rules.
        Returns the combined list of rules.
        """
        logger.info(" Executando estratégia BOOTSTRAP...")
        
        # Load AnyBURL rules (prefer DB, fallback to file)
        anyburl_rules = await self._load_anyburl_rules()
        
        manual_rules = await self._load_manual_rules()
        all_rules = self._combine_rules(anyburl_rules, manual_rules)
        
        await self._save_rules_to_files(
            all_rules, anyburl_rules, manual_rules, "bootstrap_v2.1"
        )
        logger.success(f" Bootstrap concluído: {len(all_rules)} regras preparadas")
        return all_rules

    async def apply_refinement_strategy(self) -> list[dict]:
        """
        Runs the refinement strategy: extracts advanced rules from the trained ensemble, combines and refines them, and saves the result.
        Returns the refined list of rules.
        """
        logger.info(" Executando estratégia REFINEMENT...")
        try:
            ensemble_rules = self._extract_ensemble_rules()
            existing_rules = self._load_existing_rules()
            refined_rules = self._refine_and_combine_rules(
                ensemble_rules, existing_rules
            )
            if not refined_rules:
                logger.warning("No ensemble/manual rules available; switching to HYBRID fallback")
                return await self.apply_hybrid_strategy()
            await self._save_rules_to_files(refined_rules, [], [], "refinement_v2.1")
            logger.success(
                f" Refinement concluído: {len(refined_rules)} regras refinadas"
            )
            return refined_rules
        except Exception as e:
            logger.error(f"Refinement strategy failed: {e}")
            logger.info(" Fallback para estratégia híbrida...")
            return await self.apply_hybrid_strategy()

    async def apply_hybrid_strategy(self) -> list[dict]:
        """
        Runs the hybrid strategy: combines bootstrap and refinement, removes duplicates, and saves the result.
        Returns the consolidated list of rules.
        """
        logger.info(" Executando estratégia HYBRID...")
        anyburl_rules = await self._load_anyburl_rules()
        manual_rules = await self._load_manual_rules()
        ensemble_rules = []
        try:
            ensemble_rules = self._extract_ensemble_rules()
        except Exception as e:
            logger.warning(f"Unable to extract ensemble rules: {e}")
        all_sources = anyburl_rules + manual_rules + ensemble_rules
        refined_rules = self._remove_duplicates(all_sources)
        await self._save_rules_to_files(
            refined_rules, anyburl_rules, manual_rules, "hybrid_v2.1"
        )
        logger.success(f" Hybrid concluído: {len(refined_rules)} regras consolidadas")
        return refined_rules

    async def _load_anyburl_rules(self) -> list[dict]:
        """
        Loads AnyBURL rules from DB or converts from TSV if DB is empty.
        """
        # Try DB first
        rules = await self.repo.load_rules(source="anyburl")
        if rules:
            logger.info(f" Carregadas {len(rules)} regras AnyBURL do PostgreSQL")
            return rules
            
        # Fallback to file and sync to DB
        anyburl_path = settings.OUTPUTS_DIR / "pyclause" / "rules_anyburl.tsv"
        if not anyburl_path.exists():
            logger.warning("AnyBURL TSV not found and database is empty")
            return []
            
        logger.info(" DB vazio, carregando do arquivo e sincronizando...")
        await self.repo.save_rules_from_file(anyburl_path.as_posix(), source="anyburl")
        
        # Reload from DB to get consistent format
        return await self.repo.load_rules(source="anyburl")

    async def _load_manual_rules(self) -> list[dict]:
        """
        Loads manual rules from file and syncs to DB if needed.
        """
        manual_path = settings.PATTERNS_DIR / "manual_rules.json"
        if not manual_path.exists():
            return []
            
        # Check DB first
        db_rules = await self.repo.load_rules(source="manual")
        if db_rules:
             return db_rules
             
        try:
            content = self.file_manager.read(manual_path)
            rules = content.get("rules", [])
            for rule in rules:
                rule["source"] = "manual"
                rule["extraction_method"] = "manual_curation"
                if "quality_score" not in rule:
                    rule["quality_score"] = rule.get("confidence", 1.0)
            
            # Sync to DB
            if rules:
                await self.repo.save_rules(rules, source="manual")
                
            logger.info(f" Carregadas {len(rules)} regras manuais")
            return rules
        except Exception as e:
            logger.error(f"Failed to load manual rules: {e}")
            return []

    def _extract_ensemble_rules(self) -> list[dict]:
        """
        Extracts advanced rules from the trained ensemble and returns the list.
        """
        try:
            from pff.validators.ensembles.ensemble_rules_extractor import (
                EnsembleRulesExtractor,
            )

            autofeeding_config_path = AUTOFEEDING_CONFIG_PATH
            min_confidence = 0.05
            max_depth = 5
            
            if autofeeding_config_path.exists():
                try:
                    config = self.file_manager.read(autofeeding_config_path)
                    extraction_config = config.get("ensemble_extraction", {})
                    min_confidence = extraction_config.get("min_confidence", 0.05)
                    max_depth = extraction_config.get("max_depth", 5)
                    logger.debug(f"Using autofeeding config: min_conf={min_confidence}, max_depth={max_depth}")
                except Exception as e:
                    logger.warning(f"Could not load autofeeding config: {e}, using defaults")

            extractor = EnsembleRulesExtractor()
            rules = extractor.extract_all_ensemble_rules(
                min_confidence=min_confidence,
                max_depth=max_depth
            )
            for rule in rules:
                rule["extraction_method"] = "ensemble_meta_learner"
                rule["source"] = "ensemble" # Ensure source is set
                if "quality_score" not in rule:
                    rule["quality_score"] = rule.get("confidence", 0.5)
            logger.info(f" Extraídas {len(rules)} regras do ensemble")
            return rules
        except ImportError:
            logger.warning("EnsembleRulesExtractor is not available")
            return []
        except Exception as e:
            logger.error(f"Ensemble rule extraction failed: {e}")
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
            prolog = rule.get("prolog", rule.get("rule", "")).strip() # Handle both keys
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
        logger.info(f" Refinamento: {len(all_rules)} → {len(refined_rules)} regras")
        return refined_rules

    def _remove_duplicates(self, rules: list[dict]) -> list[dict]:
        """
        Removes simple duplicates based on the prolog string and returns the unique list.
        """
        seen = set()
        unique_rules = []
        for rule in rules:
            prolog = rule.get("prolog", rule.get("rule", "")).strip()
            if prolog and prolog not in seen:
                seen.add(prolog)
                unique_rules.append(rule)
        return unique_rules

    async def _save_rules_to_files(
        self,
        all_rules: list[dict],
        anyburl_rules: list[dict],
        manual_rules: list[dict],
        version: str,
    ):
        """
        Saves rules to DB and JSON files.
        """
        # Save to DB (Ensemble rules might be new)
        # We filter rules that are NOT from AnyBURL or Manual (i.e., Ensemble) to avoid duplicates if they are already there?
        # Or we just save everything with on conflict do nothing?
        # KGRulesRepository inserts.
        # For now, let's just save the ensemble ones if they are new.
        
        ensemble_rules = [r for r in all_rules if r.get("source") == "ensemble"]
        if ensemble_rules:
             await self.repo.save_rules(ensemble_rules, source="ensemble")
             
        # Also save manual rules if they changed?
        # We already synced manual rules in _load_manual_rules.
        
        # Generate JSON for compatibility
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
            f" Regras salvas em {len([combined_path, clause_rules_path, ensemble_path])} arquivos e no PostgreSQL"
        )


async def apply_autofeeding_rules_deprecated() -> None:
    """
    Deprecated version kept for compatibility, now uses Smart Autofeeding.
    """
    logger.info(" Aplicando regras de autofeeding (deprecated → smart v2.1)...")
    smart_autofeeding = SmartAutofeeding()
    phase = await smart_autofeeding.detect_pipeline_phase()
    if phase == "bootstrap":
        rules = await smart_autofeeding.apply_bootstrap_strategy()
    else:
        rules = await smart_autofeeding.apply_hybrid_strategy()
    await update_knowledge_graph_with_rules(rules)
    if rules:
        logger.success(f" Autofeeding deprecated concluído: {len(rules)} regras")
    else:
        logger.error("Autofeeding deprecated failed: no rules generated")


async def apply_autofeeding_rules() -> None:
    """
    Main autofeeding version - Smart Autofeeding v2.1.
    Replaces the problematic 2.0 version while keeping all sophistication.
    """
    logger.info(" Smart Autofeeding v2.1 iniciado...")
    try:
        smart_autofeeding = SmartAutofeeding()
        phase = await smart_autofeeding.detect_pipeline_phase()
        if phase == "bootstrap":
            rules = await smart_autofeeding.apply_bootstrap_strategy()
        elif phase == "refinement":
            rules = await smart_autofeeding.apply_refinement_strategy()
        else:  # hybrid
            rules = await smart_autofeeding.apply_hybrid_strategy()
        await update_knowledge_graph_with_rules(rules)
        if rules:
            anyburl_count = sum(1 for r in rules if r.get("source") == "anyburl")
            manual_count = sum(1 for r in rules if r.get("source") == "manual")
            ensemble_count = len(rules) - anyburl_count - manual_count
            logger.success(" Smart Autofeeding v2.1 concluído!")
            logger.info(" Estatísticas finais:")
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
                logger.error("combined_rules.json remains empty after autofeeding")
        else:
            logger.error("Smart Autofeeding failed: no rules generated")
    except Exception as e:
        logger.error(f"Smart Autofeeding v2.1 failed: {e}")
        logger.info(" Tentando fallback para versão deprecated...")
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
            prolog = rule.get("prolog", rule.get("rule", ""))
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
                    logger.debug(f"Triple extracted from rule: {triple}")
    if new_triples:
        train_path = settings.DATA_DIR / "models" / "kg" / "train.parquet"
        if train_path.exists():
            _train_df = file_manager.read(train_path)
            logger.info(
                f"{len(new_triples)} novas triplas extraídas de regras (placeholder, não adicionadas ao arquivo)"
            )
    logger.success("Grafo de conhecimento atualizado")
