"""
Named Entity Mapper Module for Cross-Lingual Information Retrieval System.

Maps named entities across English and Bangla for cross-lingual matching.
"""

import logging
from typing import Dict, List, Optional, Set
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class NamedEntityMapper:
    """
    Map named entities between English and Bangla.

    Maintains a bilingual entity dictionary for:
    - Political figures
    - Geographic locations
    - Organizations
    - Other prominent entities

    Example:
        >>> mapper = NamedEntityMapper()
        >>> variants = mapper.get_entity_variations("Sheikh Hasina")
        >>> print(variants)
        ['Sheikh Hasina', 'শেখ হাসিনা', 'sheikh hasina']
    """

    def __init__(self, dictionary_path: Optional[str] = None):
        """
        Initialize the NamedEntityMapper.

        Args:
            dictionary_path: Path to custom entity dictionary JSON file
        """
        self.dictionary_path = dictionary_path

        # Entity mappings: normalized_key -> {en: [...], bn: [...]}
        self.entity_dict = {}

        # Load built-in dictionary
        self._load_builtin_dictionary()

        # Load custom dictionary if provided
        if dictionary_path and Path(dictionary_path).exists():
            self._load_custom_dictionary(dictionary_path)

        logger.info(
            f"NamedEntityMapper initialized with {len(self.entity_dict)} entities"
        )

    def _load_builtin_dictionary(self):
        """Load built-in entity mappings."""
        # Common political figures
        self._add_entity_mapping(
            en_variants=["Sheikh Hasina", "Hasina", "PM Hasina"],
            bn_variants=["শেখ হাসিনা", "হাসিনা"],
        )

        self._add_entity_mapping(
            en_variants=["Khaleda Zia", "Khaleda"],
            bn_variants=["খালেদা জিয়া", "খালেদা"],
        )

        # Geographic locations
        self._add_entity_mapping(
            en_variants=["Bangladesh", "BD"], bn_variants=["বাংলাদেশ"]
        )

        self._add_entity_mapping(en_variants=["Dhaka", "Dacca"], bn_variants=["ঢাকা"])

        self._add_entity_mapping(
            en_variants=["Chittagong", "Chattogram"], bn_variants=["চট্টগ্রাম"]
        )

        self._add_entity_mapping(en_variants=["Sylhet"], bn_variants=["সিলেট"])

        self._add_entity_mapping(en_variants=["Khulna"], bn_variants=["খুলনা"])

        self._add_entity_mapping(en_variants=["Rajshahi"], bn_variants=["রাজশাহী"])

        # Organizations
        self._add_entity_mapping(
            en_variants=["Awami League", "AL"], bn_variants=["আওয়ামী লীগ"]
        )

        self._add_entity_mapping(
            en_variants=["BNP", "Bangladesh Nationalist Party"],
            bn_variants=["বিএনপি", "বাংলাদেশ জাতীয়তাবাদী দল"],
        )

        self._add_entity_mapping(
            en_variants=["Bangladesh Cricket Board", "BCB"],
            bn_variants=["বাংলাদেশ ক্রিকেট বোর্ড", "বিসিবি"],
        )

        # Institutions
        self._add_entity_mapping(
            en_variants=["Dhaka University", "DU"], bn_variants=["ঢাকা বিশ্ববিদ্যালয়"]
        )

        self._add_entity_mapping(
            en_variants=["BUET", "Bangladesh University of Engineering and Technology"],
            bn_variants=["বুয়েট", "বাংলাদেশ প্রকৌশল বিশ্ববিদ্যালয়"],
        )

        # International figures (commonly mentioned in BD news)
        self._add_entity_mapping(
            en_variants=["Narendra Modi", "Modi", "PM Modi"],
            bn_variants=["নরেন্দ্র মোদি", "মোদি"],
        )

        logger.info("Built-in entity dictionary loaded")

    def _add_entity_mapping(self, en_variants: List[str], bn_variants: List[str]):
        """
        Add entity mapping to dictionary.

        Args:
            en_variants: List of English variants
            bn_variants: List of Bangla variants
        """
        # Create normalized key (use first English variant, lowercase)
        key = en_variants[0].lower().replace(" ", "_")

        self.entity_dict[key] = {
            "en": en_variants,
            "bn": bn_variants,
            "normalized_key": key,
        }

    def _load_custom_dictionary(self, path: str):
        """
        Load custom entity dictionary from JSON file.

        Args:
            path: Path to JSON file
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                custom_dict = json.load(f)

            for entry in custom_dict:
                en_vars = entry.get("en", [])
                bn_vars = entry.get("bn", [])

                if en_vars and bn_vars:
                    self._add_entity_mapping(en_vars, bn_vars)

            logger.info(f"Loaded custom entity dictionary from {path}")

        except Exception as e:
            logger.error(f"Error loading custom dictionary: {e}")

    def map_entity(self, entity: str, source_lang: str, target_lang: str) -> List[str]:
        """
        Map entity from source language to target language.

        Args:
            entity: Entity text
            source_lang: Source language ('en' or 'bn')
            target_lang: Target language ('en' or 'bn')

        Returns:
            List of entity variants in target language

        Example:
            >>> mapper.map_entity("Sheikh Hasina", "en", "bn")
            ['শেখ হাসিনা', 'হাসিনা']
        """
        if not entity or source_lang == target_lang:
            return []

        entity_normalized = entity.lower().strip()

        # Search in dictionary
        for key, mapping in self.entity_dict.items():
            # Check if entity matches any variant in source language
            source_variants = [v.lower() for v in mapping.get(source_lang, [])]

            if entity_normalized in source_variants or any(
                entity_normalized in v or v in entity_normalized
                for v in source_variants
            ):
                # Return target language variants
                return mapping.get(target_lang, [])

        logger.debug(f"No mapping found for entity: {entity}")
        return []

    def get_entity_variations(self, entity: str) -> List[str]:
        """
        Get all variations of an entity across both languages.

        Args:
            entity: Entity text

        Returns:
            List of all known variations

        Example:
            >>> mapper.get_entity_variations("Dhaka")
            ['Dhaka', 'Dacca', 'ঢাকা']
        """
        entity_normalized = entity.lower().strip()

        for key, mapping in self.entity_dict.items():
            # Check all variants in both languages
            all_variants_lower = [v.lower() for v in mapping.get("en", [])] + [
                v.lower() for v in mapping.get("bn", [])
            ]

            if entity_normalized in all_variants_lower or any(
                entity_normalized in v or v in entity_normalized
                for v in all_variants_lower
            ):
                # Return all variants (original case)
                return mapping.get("en", []) + mapping.get("bn", [])

        # If not found, return original
        return [entity]

    def build_entity_dictionary(self, entities_list: List[Dict[str, str]]) -> int:
        """
        Build entity dictionary from a list of entity mappings.

        Args:
            entities_list: List of dicts with 'en' and 'bn' keys

        Returns:
            Number of entities added

        Example:
            >>> entities = [
            ...     {'en': ['Dhaka'], 'bn': ['ঢাকা']},
            ...     {'en': ['Sylhet'], 'bn': ['সিলেট']}
            ... ]
            >>> mapper.build_entity_dictionary(entities)
            2
        """
        count = 0

        for entry in entities_list:
            en_vars = entry.get("en", [])
            bn_vars = entry.get("bn", [])

            if en_vars and bn_vars:
                self._add_entity_mapping(en_vars, bn_vars)
                count += 1

        logger.info(f"Built entity dictionary with {count} entities")
        return count

    def save_dictionary(self, path: str):
        """
        Save entity dictionary to JSON file.

        Args:
            path: Path to save JSON file
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Convert to list format
            entities_list = []
            for key, mapping in self.entity_dict.items():
                entities_list.append(
                    {"en": mapping.get("en", []), "bn": mapping.get("bn", [])}
                )

            with open(path, "w", encoding="utf-8") as f:
                json.dump(entities_list, f, ensure_ascii=False, indent=2)

            logger.info(f"Entity dictionary saved to {path}")

        except Exception as e:
            logger.error(f"Error saving dictionary: {e}")

    def find_entities_in_query(
        self, query: str, language: str = "en"
    ) -> List[Dict[str, any]]:
        """
        Find entities from dictionary that appear in query.

        Args:
            query: Query text
            language: Query language

        Returns:
            List of found entities with their mappings
        """
        query_lower = query.lower()
        found_entities = []

        for key, mapping in self.entity_dict.items():
            variants = mapping.get(language, [])

            for variant in variants:
                if variant.lower() in query_lower:
                    found_entities.append(
                        {"entity": variant, "language": language, "mappings": mapping}
                    )
                    break  # Only add once per entity

        return found_entities
