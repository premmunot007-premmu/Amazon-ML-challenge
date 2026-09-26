"""
Blocking/Candidate Generation Module
======================================
Generates candidate pairs for matching using multi-level blocking strategies.
Goal: Maximize recall (find all potential matches) while reducing candidates.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
from .utils import normalize_text, tokenize, levenshtein_ratio


class BlockingStrategy:
    """Multi-level blocking for generating candidate pairs."""
    
    def __init__(self, levenshtein_threshold: float = 0.6):
        """
        Args:
            levenshtein_threshold: Min similarity for fuzzy matching (0-1)
        """
        self.levenshtein_threshold = levenshtein_threshold
    
    def generate_candidates(
        self,
        source1: pd.DataFrame,
        source2: pd.DataFrame,
        source3: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Generate candidate pairs for all S1 entities.
        
        Args:
            source1: DataFrame with columns [entity_id, business_name, business_address, country]
            source2: DataFrame with same columns
            source3: DataFrame with same columns
        
        Returns:
            Dict: {s1_id: [list of s2/s3 candidate ids]}
        """
        candidates = {}
        
        # Combine S2 and S3
        source2_copy = source2.copy()
        source2_copy['source'] = 'S2'
        source3_copy = source3.copy()
        source3_copy['source'] = 'S3'
        sources_combined = pd.concat([source2_copy, source3_copy], ignore_index=True)
        
        # Process each S1 entity
        for _, s1_row in tqdm(source1.iterrows(), total=len(source1), desc="Blocking"):
            s1_id = s1_row['entity_id']
            candidates[s1_id] = []
            
            # Multi-level blocking
            candidate_ids = self._multi_level_blocking(
                s1_row, sources_combined
            )
            
            candidates[s1_id] = list(candidate_ids)
        
        return candidates
    
    def _multi_level_blocking(
        self,
        s1_row: pd.Series,
        sources: pd.DataFrame
    ) -> Set[str]:
        """Apply multiple blocking levels to find candidates."""
        candidates = set()
        
        # Level 1: Country + Token-based blocking
        country = s1_row['country']
        s1_name_tokens = tokenize(s1_row['business_name'])
        s1_addr_tokens = tokenize(s1_row['business_address'])
        
        # Filter by country
        same_country = sources[sources['country'] == country]
        
        # Level 2: Fuzzy name/address matching
        for _, s2_row in same_country.iterrows():
            s2_id = s2_row['entity_id']
            
            # Name similarity
            name_sim = levenshtein_ratio(
                s1_row['business_name'],
                s2_row['business_name']
            )
            
            # Address similarity
            addr_sim = levenshtein_ratio(
                s1_row['business_address'],
                s2_row['business_address']
            )
            
            # Token overlap
            s2_name_tokens = tokenize(s2_row['business_name'])
            s2_addr_tokens = tokenize(s2_row['business_address'])
            
            name_token_overlap = len(s1_name_tokens & s2_name_tokens) / max(
                len(s1_name_tokens | s2_name_tokens), 1
            )
            
            # Combined scoring: high name sim OR high addr sim OR high token overlap
            score = max(
                name_sim * 0.5 + name_token_overlap * 0.25,
                addr_sim * 0.3
            )
            
            # Aggressive threshold for recall
            if score >= self.levenshtein_threshold * 0.7:
                candidates.add(s2_id)
        
        # Level 3: Exact name token matches (high recall)
        for token in s1_name_tokens:
            if len(token) >= 4:  # Only long tokens (likely meaningful)
                same_token = sources[
                    sources['business_name'].str.contains(
                        token, case=False, na=False, regex=False
                    )
                ]
                candidates.update(same_token['entity_id'].tolist())
        
        return candidates


def create_candidate_pairs_dataframe(
    candidates: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Convert candidate dict to submission format DataFrame.
    
    Args:
        candidates: {s1_id: [list of s2/s3 ids]}
    
    Returns:
        DataFrame with columns [source1_entity_id, candidate_entity_ids]
    """
    data = []
    for s1_id, candidate_ids in candidates.items():
        data.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': ','.join(candidate_ids) if candidate_ids else ''
        })
    
    return pd.DataFrame(data)


def load_data(data_path: str, split: str = 'train') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load source TSV files.
    
    Args:
        data_path: Path to dataset directory (e.g., /path/to/dataset)
        split: 'train' or 'test'
    
    Returns:
        Tuple of (source1, source2, source3) DataFrames
    """
    base = f"{data_path}/{split}"
    
    source1 = pd.read_csv(f"{base}/{split}_source1.tsv", sep="\t")
    source2 = pd.read_csv(f"{base}/{split}_source2.tsv", sep="\t")
    source3 = pd.read_csv(f"{base}/{split}_source3.tsv", sep="\t")
    
    return source1, source2, source3
