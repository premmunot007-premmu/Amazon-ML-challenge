"""
Feature Engineering Module
===========================
Generates discriminative features for pairwise entity matching.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from .utils import (
    normalize_text, tokenize, get_tokens_list,
    levenshtein_ratio, jaccard_similarity,
    soundex, metaphone, token_overlap_ratio
)


class FeatureEngineer:
    """Generate features for entity pair matching."""
    
    def __init__(self):
        self.tfidf_name = None
        self.tfidf_addr = None
        self.name_vectors = None
        self.addr_vectors = None
    
    def fit_tfidf(self, texts: List[str], max_features: int = 100):
        """Fit TF-IDF vectorizer on texts."""
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            analyzer='char',
            ngram_range=(2, 3),
            lowercase=True
        )
        return vectorizer.fit(texts)
    
    def generate_pair_features(
        self,
        s1_entity: pd.Series,
        s2_entity: pd.Series
    ) -> dict:
        """
        Generate all features for an entity pair.
        
        Args:
            s1_entity: Row from source1 (entity_id, business_name, business_address, country)
            s2_entity: Row from source2/3
        
        Returns:
            Dict of feature_name: feature_value
        """
        features = {}
        
        # === NAME FEATURES ===
        name1 = s1_entity['business_name']
        name2 = s2_entity['business_name']
        
        # String similarity
        features['name_levenshtein'] = levenshtein_ratio(name1, name2)
        
        # Token-based
        tokens1 = tokenize(name1)
        tokens2 = tokenize(name2)
        features['name_jaccard'] = jaccard_similarity(tokens1, tokens2)
        features['name_token_overlap'] = token_overlap_ratio(name1, name2)
        
        # Token overlap percentage
        common_tokens = len(tokens1 & tokens2)
        all_tokens = len(tokens1 | tokens2)
        features['name_common_tokens'] = common_tokens
        features['name_all_tokens'] = all_tokens
        features['name_token_ratio'] = common_tokens / max(all_tokens, 1)
        
        # Length features
        features['name_len_ratio'] = len(normalize_text(name1)) / max(len(normalize_text(name2)), 1)
        features['name_len_diff'] = abs(len(normalize_text(name1)) - len(normalize_text(name2)))
        
        # Phonetic matching
        features['name_soundex_match'] = 1.0 if soundex(name1) == soundex(name2) else 0.0
        features['name_metaphone_match'] = 1.0 if metaphone(name1) == metaphone(name2) else 0.0
        
        # === ADDRESS FEATURES ===
        addr1 = s1_entity['business_address']
        addr2 = s2_entity['business_address']
        
        # String similarity
        features['addr_levenshtein'] = levenshtein_ratio(addr1, addr2)
        
        # Token-based
        addr_tokens1 = tokenize(addr1)
        addr_tokens2 = tokenize(addr2)
        features['addr_jaccard'] = jaccard_similarity(addr_tokens1, addr_tokens2)
        features['addr_token_overlap'] = token_overlap_ratio(addr1, addr2)
        
        # Token overlap percentage
        common_addr = len(addr_tokens1 & addr_tokens2)
        all_addr = len(addr_tokens1 | addr_tokens2)
        features['addr_common_tokens'] = common_addr
        features['addr_all_tokens'] = all_addr
        features['addr_token_ratio'] = common_addr / max(all_addr, 1)
        
        # Length
        features['addr_len_ratio'] = len(normalize_text(addr1)) / max(len(normalize_text(addr2)), 1)
        features['addr_len_diff'] = abs(len(normalize_text(addr1)) - len(normalize_text(addr2)))
        
        # === COUNTRY FEATURES ===
        features['country_match'] = 1.0 if s1_entity['country'] == s2_entity['country'] else 0.0
        
        # === COMBINED FEATURES ===
        features['combined_score'] = (
            features['name_levenshtein'] * 0.5 +
            features['addr_levenshtein'] * 0.3 +
            features['country_match'] * 0.2
        )
        
        # High confidence indicators
        features['high_name_sim'] = 1.0 if features['name_levenshtein'] > 0.9 else 0.0
        features['high_addr_sim'] = 1.0 if features['addr_levenshtein'] > 0.85 else 0.0
        features['high_token_overlap'] = 1.0 if features['name_token_ratio'] > 0.8 else 0.0
        
        return features
    
    def generate_pair_features_batch(
        self,
        source1: pd.DataFrame,
        sources_dict: dict,
        candidate_pairs: dict
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generate features for all candidate pairs.
        
        Args:
            source1: DataFrame with S1 entities
            sources_dict: {entity_id: row} for S2 and S3 combined
            candidate_pairs: {s1_id: [list of candidate ids]}
        
        Returns:
            Tuple of (features_df, feature_names)
        """
        all_features = []
        feature_names = None
        
        s1_dict = {row['entity_id']: row for _, row in source1.iterrows()}
        
        total_pairs = sum(len(cands) for cands in candidate_pairs.values())
        pbar = tqdm(total=total_pairs, desc="Feature Engineering")
        
        for s1_id, candidate_ids in candidate_pairs.items():
            s1_entity = s1_dict[s1_id]
            
            for s2_id in candidate_ids:
                if s2_id not in sources_dict:
                    pbar.update(1)
                    continue
                
                s2_entity = sources_dict[s2_id]
                features = self.generate_pair_features(s1_entity, s2_entity)
                
                features['source1_entity_id'] = s1_id
                features['source2_entity_id'] = s2_id
                features['label'] = None  # Will be filled during training
                
                if feature_names is None:
                    feature_names = list(features.keys())
                
                all_features.append(features)
                pbar.update(1)
        
        pbar.close()
        
        return pd.DataFrame(all_features), feature_names
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names (for model training)."""
        return [
            'name_levenshtein', 'name_jaccard', 'name_token_overlap',
            'name_common_tokens', 'name_all_tokens', 'name_token_ratio',
            'name_len_ratio', 'name_len_diff',
            'name_soundex_match', 'name_metaphone_match',
            'addr_levenshtein', 'addr_jaccard', 'addr_token_overlap',
            'addr_common_tokens', 'addr_all_tokens', 'addr_token_ratio',
            'addr_len_ratio', 'addr_len_diff',
            'country_match',
            'combined_score',
            'high_name_sim', 'high_addr_sim', 'high_token_overlap'
        ]
