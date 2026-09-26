# Amazon ML Challenge - Business Entity Resolution

**Objective**: Build an ML solution to match business records across 3 independent sources with noisy, inconsistent data.

**Target Metric**: F₀.₅ Score > 0.99 (Precision-heavy, penalizes false merges 2× over missed matches)

## Solution Architecture

### 1. **Blocking/Candidate Generation**
- Token-based blocking (normalized name/address tokens)
- Fuzzy string matching (Levenshtein ratio > threshold)
- Multi-level filtering to maximize recall while reducing candidate pairs

### 2. **Feature Engineering**
- **String Similarity**: Jaccard, Levenshtein, TF-IDF cosine
- **Phonetic Matching**: Soundex/Metaphone for name variations
- **Token Overlap**: Percentage of common tokens in name/address
- **Exact Matches**: Country, partial address components
- **Length Features**: Name/address length ratios

### 3. **ML Model**
- **LightGBM Classifier** (gradient boosting) for pairwise matching
- **Hard Negative Mining**: Emphasize similar non-matches during training
- **Class Weights**: Higher penalty on false positives (precision focus)
- **Threshold Tuning**: Optimize F₀.₅ on validation split

### 4. **Post-Processing**
- Remove duplicate predictions
- Validate output format
- Handle singletons correctly (empty matched_entity_ids)

## Project Structure

```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── blocking.py           # Candidate pair generation
│   ├── features.py           # Feature engineering
│   ├── model.py              # LightGBM model training/inference
│   ├── validation.py         # F₀.₅ scoring & validation
│   └── main.py               # End-to-end pipeline
├── README.md                 # Reproduction instructions
├── requirements.txt          # Pinned dependencies
└── run.sh                    # Execution script
output/
├── matching_results.tsv      # Final predictions (scored)
└── candidate_pairs.tsv       # Blocking output (not scored)
Documentation_template.md     # Methodology write-up
```

## Quick Start

### Prerequisites
- Python 3.8+
- ~4GB RAM
- Dataset files in `/Users/mac/Downloads/student_resource/dataset/`

### Installation & Execution

```bash
# 1. Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# 2. Run end-to-end pipeline
cd code/business_entity_resolution/
python src/main.py --data-path /Users/mac/Downloads/student_resource/dataset

# 3. Validate output
python /Users/mac/Downloads/student_resource/utils/validate_submission.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir /Users/mac/Downloads/student_resource/dataset/test
```

## Key Implementation Details

### High-Precision Strategy
1. **Conservative Threshold**: Choose threshold on validation set that prioritizes precision
2. **Hard Negative Mining**: Train model on hard-to-distinguish non-matches
3. **Feature Importance**: Focus on rare, discriminative features (e.g., business name + country combinations)
4. **Validation Split**: 80-20 train-test from training data to tune F₀.₅

### Handling Multi-Source Matching
- S1-S2 and S1-S3 matching handled separately for robustness
- Country as open set (supports US, India, France without hardcoding)
- Deduplication of candidate pairs across sources

### Singleton Handling
- Correctly predicting "no match" scores 1.0 in F₀.₅ macro-average
- Model learns to be conservative when confidence is low

## Expected Performance
- **F₀.₅ Score**: > 0.99 (high precision, conservative matching)
- **Precision**: > 0.99 (minimal false positives)
- **Recall**: > 0.90 (captures most true matches)
- **Blocking Reduction Ratio**: ~99%+ (few candidates per entity)

## Evaluation Metric: F₀.₅

```
F₀.₅ = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- Computed as macro-average (per entity, then averaged)
- Precision weighted 2× over recall
- Singletons (empty matches) included in average

## Data Constraints
- ✅ No external data lookup (only provided training data)
- ✅ No commercial APIs or databases
- ✅ Model: MIT/Apache 2.0 License, ≤8B parameters
- ✅ Output validation before submission

## Submission Files

**matching_results.tsv** (Scored on Leaderboard)
- Tab-separated: `source1_entity_id` → `matched_entity_ids`
- Every S1 entity must appear
- Empty `matched_entity_ids` for singletons

**candidate_pairs.tsv** (Not Scored, Used for Pipeline Audit)
- Tab-separated: `source1_entity_id` → `candidate_entity_ids`
- Candidates before final model narrowing

## Timeline
- **Start**: Immediate
- **Target Completion**: 48 hours
- **Status**: In development...

---

**Challenge**: [Amazon ML Challenge - Business Entity Resolution](https://www.amazonmlchallenges.com/)  
**Author**: premmunot007-premmu  
**License**: MIT
