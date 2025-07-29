#!/usr/bin/env python3
"""
Centralized Entity Dictionary Loader
Sprint 3: Centralize entity dictionaries in YAML; dynamic load 1 source of truth
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

class EntityLoader:
    """
    Centralized entity dictionary loader that reads from business_rules.yaml
    """
    
    def __init__(self, config_path: str = "config/business_rules.yaml"):
        self.config_path = Path(config_path)
        self._entities = None
        self._load_entities()
    
    def _load_entities(self) -> None:
        """Load entity dictionaries from YAML configuration"""
        try:
            if not self.config_path.exists():
                logger.error(f"Business rules file not found: {self.config_path}")
                self._entities = {}
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            entity_recognition = config.get('entity_recognition', {})
            self._entities = {
                'indian_states': entity_recognition.get('indian_states', []),
                'indian_regions': entity_recognition.get('indian_regions', []),
                'state_name_mappings': entity_recognition.get('state_name_mappings', {}),
                'energy_metrics': entity_recognition.get('energy_metrics', []),
                'aggregation_keywords': entity_recognition.get('aggregation_keywords', []),
                'growth_keywords': entity_recognition.get('growth_keywords', []),
                'comparison_keywords': entity_recognition.get('comparison_keywords', []),
                'month_keywords': entity_recognition.get('month_keywords', {})
            }
            
            logger.info(f"Loaded {len(self._entities['indian_states'])} states, {len(self._entities['indian_regions'])} regions from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load entity dictionaries: {str(e)}")
            self._entities = {}
    
    def reload_entities(self) -> None:
        """Reload entity dictionaries from YAML file"""
        logger.info("Reloading entity dictionaries from YAML")
        self._load_entities()
    
    def get_indian_states(self) -> List[str]:
        """Get list of Indian states"""
        return self._entities.get('indian_states', [])
    
    def get_indian_regions(self) -> List[str]:
        """Get list of Indian regions"""
        return self._entities.get('indian_regions', [])
    
    def get_state_name_mappings(self) -> Dict[str, str]:
        """Get state name mappings (lowercase to proper case)"""
        return self._entities.get('state_name_mappings', {})
    
    def get_energy_metrics(self) -> List[str]:
        """Get list of energy metrics"""
        return self._entities.get('energy_metrics', [])
    
    def get_aggregation_keywords(self) -> List[str]:
        """Get list of aggregation keywords"""
        return self._entities.get('aggregation_keywords', [])
    
    def get_growth_keywords(self) -> List[str]:
        """Get list of growth keywords"""
        return self._entities.get('growth_keywords', [])
    
    def get_comparison_keywords(self) -> List[str]:
        """Get list of comparison keywords"""
        return self._entities.get('comparison_keywords', [])
    
    def get_month_keywords(self) -> Dict[str, int]:
        """Get month keywords mapping"""
        return self._entities.get('month_keywords', {})
    
    def get_proper_state_name(self, state_lower: str) -> str:
        """Convert lowercase state name to proper case"""
        mappings = self.get_state_name_mappings()
        return mappings.get(state_lower.lower(), state_lower.title())
    
    def is_indian_state(self, text: str) -> bool:
        """Check if text is an Indian state"""
        states = self.get_indian_states()
        return text.lower() in [state.lower() for state in states]
    
    def is_indian_region(self, text: str) -> bool:
        """Check if text is an Indian region"""
        regions = self.get_indian_regions()
        return text.lower() in [region.lower() for region in regions]
    
    def is_energy_metric(self, text: str) -> bool:
        """Check if text is an energy metric"""
        metrics = self.get_energy_metrics()
        return text.lower() in [metric.lower() for metric in metrics]
    
    def is_aggregation_keyword(self, text: str) -> bool:
        """Check if text is an aggregation keyword"""
        keywords = self.get_aggregation_keywords()
        return text.lower() in [keyword.lower() for keyword in keywords]
    
    def get_month_number(self, month_name: str) -> Optional[int]:
        """Get month number from month name"""
        month_keywords = self.get_month_keywords()
        return month_keywords.get(month_name.lower())

# Global entity loader instance
@lru_cache()
def get_entity_loader() -> EntityLoader:
    """Get cached entity loader instance"""
    return EntityLoader()

def reload_entity_dictionaries() -> None:
    """Reload entity dictionaries from YAML file"""
    loader = get_entity_loader()
    loader.reload_entities()
    logger.info("Entity dictionaries reloaded successfully") 