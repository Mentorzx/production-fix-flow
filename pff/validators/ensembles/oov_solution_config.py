from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import numpy as np


@dataclass
class ExpertContribution:
    """
    Represents the contribution of an expert in an ensemble solution.

    Attributes:
        name (str): The name or identifier of the expert.
        score (float): The score assigned by the expert.
        confidence (float): The confidence level of the expert's contribution.
        reasoning (str): The reasoning or explanation provided by the expert.
        details (dict[str, Any]): Additional details about the expert's contribution.
    """
    name: str
    score: float
    confidence: float
    reasoning: str
    details: dict[str, Any]


class OOVAwareEnsembleManager:
    """
    Manages ensemble validation and OOV (Out-Of-Vocabulary) entity handling for knowledge graph triples.
    This class provides methods to analyze input data quality, recommend OOV handling strategies, 
    compute adaptive expert weights, adjust confidence scores, and generate diagnostics for validation results.
    It supports multiple OOV embedding strategies and dynamically balances contributions from symbolic, hybrid, 
    and neural experts based on input characteristics.
    Main Features:
    --------------
    - Input quality analysis: Calculates OOV ratio, entity type distribution, and data complexity.
    - OOV strategies: Supports zero vector, mean embedding, similarity-based, and type-based OOV embeddings.
    - Adaptive expert weighting: Adjusts ensemble weights based on rule violations and symbolic coverage.
    - Confidence adjustment: Modifies base confidence using data quality and expert agreement.
    - Diagnostic generation: Produces specialized messages for validation outcomes, highlighting detected issues.
    - Entity type detection: Classifies entities as UUIDs, codes, dates, numbers, or text.
    Methods:
    --------
    - analyze_input_quality(triples, entity_vocab, relation_vocab): Analyze triples and vocabularies for OOV and complexity.
    - compute_adaptive_expert_weights(input_quality, rule_violations, symbolic_coverage): Compute expert weights for ensemble.
    - generate_confidence_adjustment(base_confidence, input_quality, expert_agreement): Adjust confidence score.
    - generate_specialized_diagnostic(is_valid, dominant_expert, input_quality, rule_violations, expert_contributions): Generate diagnostic message.
    - OOV embedding strategies: _use_zero_vector, _use_mean_embedding, _use_similarity_embedding, _use_type_based_embedding.
    - Utility methods: _detect_entity_types, _calculate_complexity_score, _recommend_strategy, _assess_data_quality, _classify_entity_type.
    Attributes:
    -----------
    - entity_similarity_cache: Caches similarity computations for entities.
    - oov_strategies: Mapping of OOV strategy names to their implementations.
    - expert_weights: Predefined weights for ensemble experts under different scenarios.
    Example Usage:
    --------------
    >>> manager = OOVAwareEnsembleManager()
    >>> quality = manager.analyze_input_quality(triples, entity_vocab, relation_vocab)
    >>> weights = manager.compute_adaptive_expert_weights(quality, rule_violations, symbolic_coverage)
    >>> confidence = manager.generate_confidence_adjustment(base_confidence, quality, expert_agreement)
    >>> diagnostic = manager.generate_specialized_diagnostic(is_valid, dominant_expert, quality, rule_violations, weights)
    """
    def __init__(self):
        self.entity_similarity_cache = {}
        self.oov_strategies = {
            "zero_vector": self._use_zero_vector,
            "mean_embedding": self._use_mean_embedding,
            "similarity_based": self._use_similarity_embedding,
            "type_based": self._use_type_based_embedding,
        }

        self.expert_weights = {
            "base": {"symbolic": 0.4, "hybrid": 0.35, "neural": 0.25},
            "high_oov": {
                "symbolic": 0.6,
                "hybrid": 0.2,
                "neural": 0.2,
            },
            "few_rules": {
                "symbolic": 0.2,
                "hybrid": 0.5,
                "neural": 0.3,
            },
            "balanced": {"symbolic": 0.33, "hybrid": 0.34, "neural": 0.33},
        }

    def analyze_input_quality(
        self, triples: list[tuple], entity_vocab: dict, relation_vocab: dict
    ) -> dict[str, Any]:
        """
        Analyzes the quality of input triples with respect to entity vocabulary coverage, entity types, and data complexity.
        Args:
            triples (list[tuple]): List of triples, each represented as (head, relation, tail).
            entity_vocab (dict): Dictionary containing known entities.
            relation_vocab (dict): Dictionary containing known relations.
        Returns:
            dict[str, Any]: Dictionary containing:
                - "oov_ratio": Ratio of out-of-vocabulary (OOV) entities to total entities.
                - "oov_count": Number of OOV entities.
                - "total_entities": Total number of unique entities in the triples.
                - "entity_types": Types of entities detected.
                - "complexity_score": Calculated complexity score of the triples.
                - "recommended_strategy": Suggested strategy based on OOV ratio and complexity.
                - "data_quality": Assessed data quality based on OOV ratio and complexity score.
        """
        total_entities = set()
        oov_entities = set()

        for head, relation, tail in triples:
            total_entities.update([str(head), str(tail)])

            if str(head) not in entity_vocab:
                oov_entities.add(str(head))
            if str(tail) not in entity_vocab:
                oov_entities.add(str(tail))

        oov_ratio = len(oov_entities) / len(total_entities) if total_entities else 0
        entity_types = self._detect_entity_types(total_entities)
        complexity_score = self._calculate_complexity_score(triples, entity_types)
        strategy = self._recommend_strategy(oov_ratio, len(triples), complexity_score)

        return {
            "oov_ratio": oov_ratio,
            "oov_count": len(oov_entities),
            "total_entities": len(total_entities),
            "entity_types": entity_types,
            "complexity_score": complexity_score,
            "recommended_strategy": strategy,
            "data_quality": self._assess_data_quality(oov_ratio, complexity_score),
        }

    def _detect_entity_types(self, entities: set) -> dict[str, int]:
        """
        Detects and counts the types of entities in a given set.
        Entity types detected:
            - "uuids": Entities matching a UUID-like pattern (32 alphanumeric characters, possibly with dashes).
            - "dates": Entities containing 'T' and either 'Z', '+' or a timezone offset at the end.
            - "numbers": Entities that represent numeric values (digits, possibly with '.' or '-').
            - "codes": Entities that are alphanumeric codes (contain '_' or are single-word strings).
            - "text": Entities that do not match any of the above patterns.
        Args:
            entities (set): A set of entities to classify.
        Returns:
            dict[str, int]: A dictionary with counts for each entity type.
        """
        types = {"uuids": 0, "codes": 0, "dates": 0, "numbers": 0, "text": 0}

        for entity in entities:
            entity_str = str(entity)

            # UUID pattern
            if len(entity_str) == 32 and entity_str.replace("-", "").isalnum():
                types["uuids"] += 1
            # Date pattern
            elif "T" in entity_str and (
                "Z" in entity_str or "+" in entity_str or "-" in entity_str[-6:]
            ):
                types["dates"] += 1
            # Number pattern
            elif entity_str.replace(".", "").replace("-", "").isdigit():
                types["numbers"] += 1
            # Code pattern (alphanumeric with specific structure)
            elif "_" in entity_str or len(entity_str.split()) == 1:
                types["codes"] += 1
            else:
                types["text"] += 1

        return types

    def _calculate_complexity_score(
        self, triples: list[tuple], entity_types: dict[str, int]
    ) -> float:
        """
        Calculates a complexity score for a set of triples based on relation diversity, average degree, and entity type diversity.
        Args:
            triples (list[tuple]): A list of triples, where each triple is a tuple (head, relation, tail).
            entity_types (dict[str, int]): A dictionary mapping entity types to their counts.
        Returns:
            float: The calculated complexity score, normalized between 0 and 1.
        """
        unique_relations = len(set(relation for _, relation, _ in triples))
        avg_degree = len(triples) / max(1, sum(entity_types.values()))
        type_diversity = len([t for t in entity_types.values() if t > 0])
        relation_score = min(
            unique_relations / 20.0, 1.0
        ) 
        degree_score = min(avg_degree / 5.0, 1.0)
        diversity_score = type_diversity / 5.0

        return (relation_score + degree_score + diversity_score) / 3.0

    def _recommend_strategy(
        self, oov_ratio: float, triple_count: int, complexity: float
    ) -> str:
        """
        Recommends a strategy based on the out-of-vocabulary (OOV) ratio, the number of triples, and the complexity.

        Args:
            oov_ratio (float): The ratio of OOV items in the dataset.
            triple_count (int): The number of triples (rules) available.
            complexity (float): The complexity score of the current configuration.

        Returns:
            str: The recommended strategy, which can be one of:
                - "high_oov": If the OOV ratio is greater than 0.8.
                - "few_rules": If the triple count is less than 50.
                - "balanced": If the complexity is greater than 0.7.
                - "base": For all other cases.
        """
        if oov_ratio > 0.8:
            return "high_oov"
        elif triple_count < 50:
            return "few_rules"
        elif complexity > 0.7:
            return "balanced"
        else:
            return "base"

    def _assess_data_quality(self, oov_ratio: float, complexity: float) -> str:
        """
        Assess the quality of data based on out-of-vocabulary (OOV) ratio and complexity.

        Parameters:
            oov_ratio (float): The ratio of out-of-vocabulary items in the data.
            complexity (float): A measure of the data's complexity.

        Returns:
            str: A qualitative assessment of data quality, which can be one of
                 "excellent", "good", "fair", or "poor".
        """
        if oov_ratio < 0.2 and complexity > 0.5:
            return "excellent"
        elif oov_ratio < 0.5 and complexity > 0.3:
            return "good"
        elif oov_ratio < 0.8:
            return "fair"
        else:
            return "poor"

    def compute_adaptive_expert_weights(
        self,
        input_quality: dict[str, Any],
        rule_violations: int,
        symbolic_coverage: float,
    ) -> dict[str, float]:
        """
        Computes adaptive weights for different expert models based on input quality, rule violations, and symbolic coverage.
        The method adjusts the base weights for 'symbolic', 'hybrid', and 'neural' experts according to:
        - The recommended strategy from input_quality.
        - The number of rule violations:
            * If rule_violations > 5: increases 'symbolic' weight, decreases 'hybrid' weight.
            * If rule_violations == 0: decreases 'symbolic' weight, increases 'hybrid' weight.
        - The symbolic coverage:
            * If symbolic_coverage < 0.3: increases 'neural' weight, decreases 'symbolic' weight.
        The final weights are normalized to sum to 1.
        Args:
            input_quality (dict[str, Any]): Dictionary containing input quality metrics, including 'recommended_strategy'.
            rule_violations (int): Number of rule violations detected.
            symbolic_coverage (float): Proportion of symbolic coverage in the input.
        Returns:
            dict[str, float]: Normalized weights for each expert model.
        """
        strategy = input_quality["recommended_strategy"]
        base_weights = self.expert_weights[strategy].copy()

        if rule_violations > 5:
            base_weights["symbolic"] *= 1.2
            base_weights["hybrid"] *= 0.9
        elif rule_violations == 0:
            base_weights["symbolic"] *= 0.8
            base_weights["hybrid"] *= 1.1

        if symbolic_coverage < 0.3:
            base_weights["neural"] *= 1.2
            base_weights["symbolic"] *= 0.8

        total_weight = sum(base_weights.values())
        return {k: v / total_weight for k, v in base_weights.items()}

    def generate_confidence_adjustment(
        self,
        base_confidence: float,
        input_quality: dict[str, Any],
        expert_agreement: float,
    ) -> float:
        """
        Adjusts the base confidence score based on input data quality, out-of-vocabulary (OOV) ratio, and expert agreement.
        Parameters:
            base_confidence (float): The initial confidence score to be adjusted.
            input_quality (dict[str, Any]): Dictionary containing input quality metrics:
                - 'data_quality' (str): Quality of the input data ('excellent', 'good', 'fair', 'poor').
                - 'oov_ratio' (float): Ratio of out-of-vocabulary terms in the input.
            expert_agreement (float): Level of agreement among experts, ranging from 0.0 to 1.0.
        Returns:
            float: The adjusted confidence score, constrained between 0.1 and 0.99.
        """
        quality_modifier = {"excellent": 1.0, "good": 0.95, "fair": 0.85, "poor": 0.7}

        data_quality = input_quality["data_quality"]
        oov_ratio = input_quality["oov_ratio"]

        confidence = base_confidence * quality_modifier.get(data_quality, 0.7)

        if oov_ratio > 0.6:
            confidence *= 1 - (
                oov_ratio - 0.6
            )
        if expert_agreement < 0.5:
            confidence *= 0.8
        elif expert_agreement > 0.8:
            confidence *= 1.1

        return min(max(confidence, 0.1), 0.99)

    def generate_specialized_diagnostic(
        self,
        is_valid: bool,
        dominant_expert: str,
        input_quality: dict[str, Any],
        rule_violations: list[dict],
        expert_contributions: dict[str, float],
    ) -> str:
        """
        Generates a specialized diagnostic message based on the validity of the input and various quality metrics.
        Args:
            is_valid (bool): Indicates whether the input passes validation.
            dominant_expert (str): The name of the expert with the most influence in the ensemble.
            input_quality (dict[str, Any]): Dictionary containing input quality metrics, including 'data_quality' and 'oov_ratio'.
            rule_violations (list[dict]): List of rule violations detected during validation.
            expert_contributions (dict[str, float]): Mapping of expert names to their contribution scores.
        Returns:
            str: A diagnostic message summarizing the validation outcome and relevant details.
        """
        data_quality = input_quality["data_quality"]
        oov_ratio = input_quality["oov_ratio"]

        if is_valid:
            return self._generate_positive_diagnostic(
                dominant_expert, data_quality, expert_contributions
            )
        else:
            return self._generate_negative_diagnostic(
                dominant_expert,
                data_quality,
                oov_ratio,
                rule_violations,
                expert_contributions,
            )

    def _generate_positive_diagnostic(
        self, dominant_expert: str, data_quality: str, contributions: dict[str, float]
    ) -> str:
        """
        Generates a positive diagnostic message based on the dominant expert, data quality, and expert contributions.
        Args:
            dominant_expert (str): The key representing the dominant expert ('symbolic', 'hybrid', 'neural', etc.).
            data_quality (str): The quality of the data ('excellent', 'good', 'fair', 'poor').
            contributions (dict[str, float]): A dictionary mapping expert names to their contribution weights.
        Returns:
            str: A formatted diagnostic message describing the data quality and the dominant expert's contribution.
        """
        quality_msgs = {
            "excellent": "Dados de alta qualidade validados",
            "good": "Dados bem estruturados aprovados",
            "fair": "Dados aceitáveis após análise robusta",
            "poor": "Dados validados apesar da qualidade limitada",
        }
        base_msg = quality_msgs.get(data_quality, "Dados validados")
        expert_details = {
            "symbolic": f"por regras de negócio (peso: {contributions.get('symbolic', 0):.1%})",
            "hybrid": f"por análise híbrida neural-simbólica (peso: {contributions.get('hybrid', 0):.1%})",
            "neural": f"por padrões neurais (peso: {contributions.get('neural', 0):.1%})",
        }

        detail = expert_details.get(dominant_expert, "pelo ensemble completo")
        return f"{base_msg} {detail}"

    def _generate_negative_diagnostic(
        self,
        dominant_expert: str,
        data_quality: str,
        oov_ratio: float,
        violations: list[dict],
        contributions: dict[str, float],
    ) -> str:
        """
        Generates a diagnostic message describing negative findings based on the dominant expert,
        data quality, out-of-vocabulary (OOV) ratio, rule violations, and expert contributions.
        Parameters:
            dominant_expert (str): The type of expert that dominated the decision ("symbolic", "hybrid", "neural", or "ensemble").
            data_quality (str): The assessed quality of the input data ("poor", etc.).
            oov_ratio (float): The ratio of out-of-vocabulary entities detected.
            violations (list[dict]): List of rule violation details, each as a dictionary.
            contributions (dict[str, float]): Mapping of expert names to their contribution scores.
        Returns:
            str: A diagnostic message summarizing the detected negative findings.
        """
        if dominant_expert == "symbolic" and violations:
            violation_types = [v.get("field", "campo") for v in violations[:2]]
            return f"Violação de regras de negócio detectada em: {', '.join(violation_types)}"
        elif dominant_expert == "hybrid" or dominant_expert == "neural":
            if oov_ratio > 0.7:
                return f"Padrões estruturais anômalos detectados (entidades desconhecidas: {oov_ratio:.1%})"
            else:
                return "Inconsistências identificadas por análise de padrões neurais"
        else:  # ensemble
            problems = []
            if violations:
                problems.append(f"{len(violations)} violações de regras")
            if oov_ratio > 0.5:
                problems.append(
                    f"alta proporção de entidades desconhecidas ({oov_ratio:.1%})"
                )
            if data_quality == "poor":
                problems.append("qualidade de dados comprometida")

            if problems:
                return f"Múltiplos problemas detectados: {', '.join(problems)}"
            else:
                return "Inconsistências detectadas pelo ensemble de especialistas"

    def _use_zero_vector(self, entity: str, embedding_dim: int) -> np.ndarray:
        """
        Returns a zero vector of the specified embedding dimension.

        Args:
            entity (str): The entity for which the zero vector is generated.
            embedding_dim (int): The dimension of the embedding vector.

        Returns:
            np.ndarray: A numpy array of zeros with shape (embedding_dim,).
        """
        return np.zeros(embedding_dim)

    def _use_mean_embedding(self, entity: str, embeddings: np.ndarray) -> np.ndarray:
        """
        Computes the mean embedding vector for a given entity.

        Args:
            entity (str): The name or identifier of the entity.
            embeddings (np.ndarray): An array of embedding vectors associated with the entity.

        Returns:
            np.ndarray: The mean embedding vector computed across all provided embeddings.
        """
        return np.mean(embeddings, axis=0)

    def _use_similarity_embedding(
        self, entity: str, entity_vocab: dict, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Generates an embedding for an out-of-vocabulary (OOV) entity by finding similar entities in the vocabulary
        and combining their embeddings using a weighted average and noise for variability.
        Args:
            entity (str): The entity for which to generate an embedding.
            entity_vocab (dict): A dictionary mapping known entities to their indices in the embeddings array.
            embeddings (np.ndarray): An array of embeddings corresponding to the entities in the vocabulary.
        Returns:
            np.ndarray: The generated embedding for the OOV entity.
        Method:
            - Computes similarity scores between the OOV entity and known entities based on length, prefix, suffix,
              entity type, and string similarity.
            - Selects the top similar entities and combines their embeddings using a weighted average.
            - Adds random noise to the resulting embedding for variability.
            - If no sufficiently similar entities are found, returns the average of all embeddings and the most frequent
              embedding, with added noise.
        """
        def string_similarity(a, b):
            return SequenceMatcher(None, a, b).ratio()

        candidates = []
        entity_str = str(entity)
        entity_len = len(entity_str)
        entity_prefix = entity_str[:4]
        entity_suffix = entity_str[-4:]
        entity_type = self._classify_entity_type(entity_str)

        for e in entity_vocab.keys():
            e_str = str(e)
            score = 0
            score += 0.2 * (1 - abs(len(e_str) - entity_len) / max(1, entity_len))
            score += 0.2 * (1 if e_str.startswith(entity_prefix) else 0)
            score += 0.2 * (1 if e_str.endswith(entity_suffix) else 0)
            score += 0.2 * (
                1 if self._classify_entity_type(e_str) == entity_type else 0
            )
            score += 0.2 * string_similarity(entity_str, e_str)
            candidates.append((score, e_str))

        candidates.sort(reverse=True)
        top_k = [e for s, e in candidates[:3] if s > 0.5]
        indices = [entity_vocab[e] for e in top_k if e in entity_vocab]

        if indices:
            selected_embeddings = embeddings[indices]
            weights = np.linspace(1.0, 0.7, num=len(selected_embeddings))
            weighted_emb = np.average(selected_embeddings, axis=0, weights=weights)
            noise = np.random.normal(0, 0.03, weighted_emb.shape)
            return weighted_emb + noise
        else:
            mean_emb = np.mean(embeddings, axis=0)
            freq_idx = np.argmax(np.bincount(list(entity_vocab.values())))
            freq_emb = embeddings[freq_idx]
            noise = np.random.normal(0, 0.05, mean_emb.shape)
            return (mean_emb + freq_emb) / 2 + noise

    def _use_type_based_embedding(
        self, entity: str, entity_types: dict, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Computes an embedding for a given entity based on its type.
        If there are entities of the same type as the input entity, their embeddings are averaged.
        Otherwise, returns the mean of all embeddings with added Gaussian noise.
        Args:
            entity (str): The entity for which to compute the embedding.
            entity_types (dict): A mapping from entity types to lists of entities.
            embeddings (np.ndarray): Array of embeddings corresponding to entities.
        Returns:
            np.ndarray: The computed embedding for the entity.
        """
        entity_type = self._classify_entity_type(entity)
        dim = embeddings.shape[1]

        same_type_entities = [
            e
            for e in entity_types.get(entity_type, [])
            if e in entity_types.get(entity_type, [])
        ]
        if same_type_entities:
            indices = [
                i
                for i, e in enumerate(entity_types.get(entity_type, []))
                if e in same_type_entities
            ]
            if indices:
                selected_embeddings = embeddings[indices]
                return np.mean(selected_embeddings, axis=0)
        mean_emb = np.mean(embeddings, axis=0)
        noise = np.random.normal(0, 0.05, dim)
        return mean_emb + noise

    def _classify_entity_type(self, entity: str) -> str:
        """
        Classifies the type of a given entity based on its string representation.

        Args:
            entity (str): The entity to classify.

        Returns:
            str: The type of the entity, which can be one of the following:
                - "uuid": if the entity string has a length of 32 characters.
                - "number": if the entity string consists only of digits.
                - "datetime": if the entity string contains the character 'T'.
                - "text": for all other cases.
        """
        entity_str = str(entity)
        if len(entity_str) == 32:
            return "uuid"
        elif entity_str.isdigit():
            return "number"
        elif "T" in entity_str:
            return "datetime"
        else:
            return "text"


class EnhancedEnsembleConfig:
    """
    Configuration class for enhanced ensemble validation strategies.
    Attributes:
        oov_strategy (str): Strategy for handling out-of-vocabulary (OOV) entities. Default is "similarity_based".
        confidence_threshold (float): Minimum confidence required for ensemble decisions. Default is 0.6.
        min_rule_coverage (float): Minimum coverage required for rules to be considered. Default is 0.3.
        expert_diversity_bonus (float): Bonus factor for diversity among experts in the ensemble. Default is 0.1.
        diagnostic_thresholds (dict): Thresholds for categorizing confidence levels (high, medium, low).
        diagnostic_templates (dict): Templates for diagnostic messages, supporting rule violations, pattern anomalies,
            OOV degradation, and ensemble conflicts.
    """

    def __init__(self):
        self.oov_strategy = "similarity_based"
        self.confidence_threshold = 0.6
        self.min_rule_coverage = 0.3
        self.expert_diversity_bonus = 0.1
        self.diagnostic_thresholds = {
            "high_confidence": 0.8,
            "medium_confidence": 0.6,
            "low_confidence": 0.4,
        }
        self.diagnostic_templates = {
            "rule_violation": "Regra violada: {rule_description} no campo {field}",
            "pattern_anomaly": "Padrão anômalo detectado em {entity_type} com {confidence:.1%} de confiança",
            "oov_degradation": "Qualidade reduzida devido a {oov_count} entidades desconhecidas",
            "ensemble_conflict": "Conflito entre especialistas: {conflicts}",
        }


def apply_enhanced_analysis(
    triples: list[tuple],
    prediction_proba: np.ndarray,
    entity_vocab: dict,
    relation_vocab: dict,
    rule_violations: list[dict],
) -> dict[str, Any]:
    """
    Performs an enhanced analysis of prediction results using an OOV-aware ensemble manager.
    This function evaluates the quality of input triples, computes adaptive expert weights,
    assesses symbolic coverage and expert agreement, and adjusts prediction confidence accordingly.
    It also generates a specialized diagnostic and recommendations based on the analysis.
    Args:
        triples (list[tuple]): List of input triples to be analyzed.
        prediction_proba (np.ndarray): Array of prediction probabilities for each class.
        entity_vocab (dict): Dictionary mapping entity identifiers to their representations.
        relation_vocab (dict): Dictionary mapping relation identifiers to their representations.
        rule_violations (list[dict]): List of rule violation records, each as a dictionary.
    Returns:
        dict[str, Any]: A dictionary containing:
            - "original_confidence": The original maximum prediction confidence.
            - "adjusted_confidence": The confidence value adjusted by input quality and expert agreement.
            - "dominant_expert": The expert with the highest adaptive weight.
            - "expert_weights": Dictionary of computed weights for each expert.
            - "input_quality": Quality assessment of the input triples.
            - "specialized_diagnostic": Diagnostic information generated by the ensemble manager.
            - "expert_agreement": Measure of agreement among experts.
            - "recommendations": List of recommendations based on input quality and expert weights.
    """
    manager = OOVAwareEnsembleManager()
    input_quality = manager.analyze_input_quality(triples, entity_vocab, relation_vocab)
    symbolic_coverage = len(
        [v for v in rule_violations if v.get("applied", False)]
    ) / max(1, len(rule_violations))
    expert_weights = manager.compute_adaptive_expert_weights(
        input_quality, len(rule_violations), symbolic_coverage
    )
    dominant_expert = max(expert_weights, key=expert_weights.get)
    weight_values = list(expert_weights.values())
    expert_agreement = 1.0 - (max(weight_values) - min(weight_values))
    base_confidence = float(prediction_proba.max())
    adjusted_confidence = manager.generate_confidence_adjustment(
        base_confidence, input_quality, expert_agreement
    )
    is_valid = prediction_proba.argmax() == 1  # Assumindo que classe 1 é válido
    diagnostic = manager.generate_specialized_diagnostic(
        is_valid, dominant_expert, input_quality, rule_violations, expert_weights
    )

    return {
        "original_confidence": base_confidence,
        "adjusted_confidence": adjusted_confidence,
        "dominant_expert": dominant_expert,
        "expert_weights": expert_weights,
        "input_quality": input_quality,
        "specialized_diagnostic": diagnostic,
        "expert_agreement": expert_agreement,
        "recommendations": _generate_recommendations(input_quality, expert_weights),
    }


def _generate_recommendations(
    input_quality: dict[str, Any], expert_weights: dict[str, float]
) -> list[str]:
    """
    Generates a list of recommendations based on input data quality and expert model weights.
    Args:
        input_quality (dict[str, Any]): Dictionary containing quality metrics of the input data, such as 'oov_ratio' and 'data_quality'.
        expert_weights (dict[str, float]): Dictionary with weights assigned to different expert models, e.g., 'symbolic' and 'neural'.
    Returns:
        list[str]: A list of recommendation strings tailored to the provided input quality and expert weights.
    """
    recommendations = []
    if input_quality["oov_ratio"] > 0.6:
        recommendations.append(
            "Considere expandir o vocabulário de entidades do modelo TransE"
        )
    if input_quality["data_quality"] == "poor":
        recommendations.append("Recomenda-se revisão da qualidade dos dados de entrada")
    if expert_weights["symbolic"] > 0.6:
        recommendations.append("Foco em validação de regras de negócio recomendado")
    if expert_weights["neural"] > 0.5:
        recommendations.append("Análise de padrões estruturais pode ser relevante")

    return recommendations


# Exemplo de uso:
"""
# No business_service.py, seria usado assim:

manager = OOVAwareEnsembleManager()
enhanced_result = apply_enhanced_analysis(
    triples=converted_triples,
    prediction_proba=prediction_proba,
    entity_vocab=entity_to_idx,
    relation_vocab=relation_to_idx,
    rule_violations=rule_analysis["violated_rules"]
)

# Resultado teria informações muito mais específicas:
print(f"Expert dominante: {enhanced_result['dominant_expert']}")
print(f"Qualidade dos dados: {enhanced_result['input_quality']['data_quality']}")
print(f"Diagnóstico: {enhanced_result['specialized_diagnostic']}")
print(f"Pesos dos especialistas: {enhanced_result['expert_weights']}")
"""
