from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any
from pydantic import BaseModel


@dataclass
class FeatureInfo:
    """Information about a specific feature."""

    name: str
    source_field: str
    transformation: str
    index: int
    category: str
    description: str | None = None


class FeatureMapper:
    """Maps numeric features to Pydantic model fields maintaining traceability."""

    def __init__(self):
        self.feature_map: dict[int, FeatureInfo] = {}
        self.field_to_features: dict[str, list[int]] = {}
        self.current_index = 0

    def register_feature(
        self,
        source_field: str,
        transformation: str,
        category: str,
        description: str | None = None,
    ) -> int:
        """Register a new feature and return its index."""
        feature_info = FeatureInfo(
            name=f"{category}_{source_field}_{transformation}",
            source_field=source_field,
            transformation=transformation,
            index=self.current_index,
            category=category,
            description=description or f"{category} feature for {source_field}",
        )

        self.feature_map[self.current_index] = feature_info

        if source_field not in self.field_to_features:
            self.field_to_features[source_field] = []
        self.field_to_features[source_field].append(self.current_index)

        self.current_index += 1
        return self.current_index - 1

    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names."""
        return [self.feature_map[i].name for i in range(len(self.feature_map))]

    def get_feature_info(self, index: int) -> FeatureInfo | None:
        """Return information about a specific feature."""
        return self.feature_map.get(index)

    def get_readable_explanation(
        self, feature_indices: list[int], impacts: list[float]
    ) -> str:
        """Generate human-readable explanation of top features."""
        explanations = []

        for idx, impact in zip(feature_indices[:3], impacts[:3]):
            feature_info = self.get_feature_info(idx)
            if feature_info:
                direction = "increased" if impact > 0 else "decreased"
                field_name = feature_info.source_field.replace("_", " ").title()
                explanations.append(
                    f"{field_name} ({feature_info.transformation}) {direction} validation confidence"
                )

        return f"Key factors: {'; '.join(explanations)}"

    def save(self, path: str | Path) -> None:
        """Save mapping to file."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "feature_map": self.feature_map,
                    "field_to_features": self.field_to_features,
                    "current_index": self.current_index,
                },
                f,
            )

    def load(self, path: str | Path) -> None:
        """Load mapping from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.feature_map = data["feature_map"]
            self.field_to_features = data["field_to_features"]
            self.current_index = data["current_index"]


class GenericFeatureExtractor:
    """Generic feature extractor that works with any Pydantic model."""

    def __init__(self, target_features: int = 324):
        self.target_features = target_features
        self.mapper = FeatureMapper()

    def build_mapping_from_model(self, model_class: type[BaseModel]) -> None:
        """Build feature mapping dynamically from Pydantic model structure."""
        self._register_base_features()
        self._register_model_fields(model_class)
        self._pad_to_target_features()

    def _register_base_features(self) -> None:
        """Register basic features that work with any data structure."""
        self.mapper.register_feature("msisdn", "length", "basic", "MSISDN digit count")
        self.mapper.register_feature(
            "msisdn", "country_code", "basic", "Country code presence"
        )
        self.mapper.register_feature(
            "msisdn", "area_code", "basic", "Area code analysis"
        )
        self.mapper.register_feature("msisdn", "digit_sum", "basic", "Sum of digits")
        self.mapper.register_feature("msisdn", "zero_count", "basic", "Count of zeros")

        self.mapper.register_feature("id", "hash", "basic", "ID hash value")
        self.mapper.register_feature("externalId", "hash", "basic", "External ID hash")
        self.mapper.register_feature("status", "hash", "basic", "Status hash")

    def _register_model_fields(self, model_class: type[BaseModel]) -> None:
        """Register features based on model field structure."""
        schema = model_class.model_json_schema()
        properties = schema.get("properties", {})

        for field_name, field_schema in properties.items():
            self._register_field_features(field_name, field_schema)

    def _register_field_features(
        self, field_name: str, field_schema: dict[str, Any]
    ) -> None:
        """Register features for a specific field based on its schema."""
        field_type = field_schema.get("type", "unknown")

        self.mapper.register_feature(
            field_name, "presence", "structural", f"Presence of {field_name}"
        )

        if field_type == "array":
            self.mapper.register_feature(
                field_name, "length", "structural", f"Length of {field_name} array"
            )
            self.mapper.register_feature(
                field_name, "empty", "structural", f"Empty {field_name} indicator"
            )

            items_schema = field_schema.get("items", {})
            if items_schema.get("type") == "object":
                self._register_nested_object_features(field_name, items_schema)

        elif field_type == "object":
            self.mapper.register_feature(
                field_name, "depth", "structural", f"Nesting depth of {field_name}"
            )
            self._register_nested_object_features(field_name, field_schema)

        elif field_type == "string":
            self.mapper.register_feature(
                field_name, "string_length", "content", f"String length of {field_name}"
            )
            self.mapper.register_feature(
                field_name, "string_hash", "content", f"String hash of {field_name}"
            )

    def _register_nested_object_features(
        self, parent_field: str, object_schema: dict[str, Any]
    ) -> None:
        """Register features for nested object properties."""
        properties = object_schema.get("properties", {})

        for nested_field, nested_schema in properties.items():
            nested_path = f"{parent_field}.{nested_field}"
            nested_type = nested_schema.get("type", "unknown")

            self.mapper.register_feature(
                nested_path, "presence", "nested", f"Presence of {nested_path}"
            )

            if nested_type == "array":
                self.mapper.register_feature(
                    nested_path, "count", "nested", f"Count in {nested_path}"
                )

    def _pad_to_target_features(self) -> None:
        """Pad with computed features to reach target count."""
        current_count = len(self.mapper.feature_map)
        remaining = self.target_features - current_count

        for i in range(remaining):
            self.mapper.register_feature(
                "computed",
                f"feature_{i}",
                "padding",
                f"Computed feature {i} for model completeness",
            )

    def extract_features(self, data: dict[str, Any]) -> tuple[list[float], list[str]]:
        """Extract features from data maintaining mapping."""
        features = []

        features.extend(self._extract_basic_features(data))
        features.extend(self._extract_structural_features(data))
        features.extend(self._extract_content_features(data))
        features.extend(self._extract_computed_features(data))

        while len(features) < self.target_features:
            features.append(0.0)

        features = features[: self.target_features]
        feature_names = self.mapper.get_feature_names()[: len(features)]

        return features, feature_names

    def _extract_basic_features(self, data: dict[str, Any]) -> list[float]:
        """Extract basic features that work with any data."""
        features = []

        msisdn = str(data.get("msisdn", ""))
        if msisdn:
            features.extend(
                [
                    len(msisdn) / 15.0,  # normalized length
                    float(msisdn.startswith("55")),  # country code
                    float(msisdn[:3] in ["551", "552"])
                    if len(msisdn) >= 3
                    else 0.0,  # area code
                    sum(int(d) for d in msisdn if d.isdigit()) / 100.0,  # digit sum
                    msisdn.count("0") / len(msisdn) if msisdn else 0.0,  # zero ratio
                ]
            )
        else:
            features.extend([0.0] * 5)

        features.append(float(hash(str(data.get("id", ""))) % 1000) / 1000.0)
        features.append(float(hash(str(data.get("externalId", ""))) % 1000) / 1000.0)
        features.append(float(hash(str(data.get("status", ""))) % 1000) / 1000.0)

        return features

    def _extract_structural_features(self, data: dict[str, Any]) -> list[float]:
        """Extract features based on data structure."""
        features = []

        for key, value in data.items():
            features.append(1.0)  # presence

            if isinstance(value, list):
                features.append(len(value) / 10.0)  # normalized length
                features.append(float(len(value) == 0))  # empty indicator

                if value and isinstance(value[0], dict):
                    avg_keys = sum(
                        len(item.keys()) for item in value if isinstance(item, dict)
                    ) / len(value)
                    features.append(avg_keys / 20.0)  # normalized average keys
                else:
                    features.append(0.0)

            elif isinstance(value, dict):
                features.append(len(value) / 20.0)  # normalized depth
                features.append(
                    self._get_nesting_depth(value) / 5.0
                )  # normalized nesting
            else:
                features.extend([0.0, 0.0])

        target_structural = 50
        while len(features) < target_structural:
            features.append(0.0)

        return features[:target_structural]

    def _extract_content_features(self, data: dict[str, Any]) -> list[float]:
        """Extract content-based features."""
        features = []

        for key, value in data.items():
            if isinstance(value, str):
                features.append(len(value) / 100.0)  # string length
                features.append(float(hash(value) % 1000) / 1000.0)  # string hash
            elif isinstance(value, (int, float)):
                features.append(
                    float(value) / 1000.0 if value < 10000 else 1.0
                )  # normalized number
                features.append(0.0)  # placeholder
            else:
                features.extend([0.0, 0.0])

        target_content = 40
        while len(features) < target_content:
            features.append(0.0)

        return features[:target_content]

    def _extract_computed_features(self, data: dict[str, Any]) -> list[float]:
        """Extract computed/derived features."""
        features = []

        contract_count = len(data.get("contract", []))
        account_count = len(data.get("account", []))
        status_count = len(data.get("status", []))

        features.extend(
            [
                contract_count / 5.0,  # normalized contract count
                account_count / 3.0,  # normalized account count
                status_count / 5.0,  # normalized status count
                float(
                    contract_count > 0 and account_count > 0
                ),  # has both contract and account
            ]
        )

        if "contract" in data and data["contract"]:
            first_contract = (
                data["contract"][0]
                if isinstance(data["contract"], list)
                else data["contract"]
            )
            if isinstance(first_contract, dict):
                product_count = len(first_contract.get("product", []))
                features.append(product_count / 10.0)  # normalized product count
            else:
                features.append(0.0)
        else:
            features.append(0.0)

        remaining = self.target_features - 8 - 50 - 40 - len(features)
        features.extend([0.0] * remaining)

        return features

    def _get_nesting_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of a data structure."""
        if not isinstance(obj, (dict, list)) or current_depth > 10:
            return current_depth

        if isinstance(obj, dict):
            return max(
                (self._get_nesting_depth(v, current_depth + 1) for v in obj.values()),
                default=current_depth,
            )
        elif isinstance(obj, list) and obj:
            return max(
                (self._get_nesting_depth(item, current_depth + 1) for item in obj),
                default=current_depth,
            )

        return current_depth
