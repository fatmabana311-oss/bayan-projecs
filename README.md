# Final-Project--Bayan-bilingual-citizen-feedback
# Project Overview: 
This is an AI-powered system designed to process citizen complaints and suggestions. It serves municipal operations by automatically sorting incoming text requests into administrative categories and allowing agents to immediately find historically resolved cases that match new incoming issues—even if they are written in a different language or dialect.
# Problem Description: 
Municipal operational centers deal with large volumes of unstructured, multi-dialectal, and bilingual feedback daily. Operational bottlenecks include:
1. **Topic Classification**: Inconsistent classification of incoming complaints due to dialectal nuances (e.g., Gulf vs. Egyptian Arabic vs. MSA) and varying feedback lengths.
2. **Historical Case Retrieval**: The inability of customer service agents to quickly locate previous resolutions for similar historical issues, resulting in slow and disjointed manual routing.
3. **Ambiguity and Noise**: Short, vague, or multi-topic messages that cannot be accurately categorized by simple rule-based parser
# Dataset & Model Used:
- **Dataset**: 
  - **`bayan_feedback_clean.csv`**: Over 3,400 cleaned, labeled, and annotated feedback items.
  - **`bayan_cases.csv`**: A repository of 20,000 resolved historical cases used for building the semantic search database.
  - **`bayan_queries.jsonl`**: Standardized, human-judged test queries used to evaluate search metrics.
- **Models Used**:
  - **Classifier**: **XLM-RoBERTa (fine-tuned)** fine-tuned locally for multi-class classification (`bayan_topic_model`).
  - **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to support cross-lingual semantic matching (English $\leftrightarrow$ Arabic).
  - **Re-ranker**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` for deep context alignment.
# Workflow / Architecture:
     [ Incoming Feedback ]
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
 [ XLM-RoBERTa Classifier ]   [ Multilingual Sentence Transformer ]
          │                           │
          ▼                           ▼
 [ Topic Prediction ]        [ L2 Normalized Embeddings ]
                                      │
                                      ▼
                           [ FAISS Indexing (FlatIP) ]
                                      │
                                      ▼
                           [ Cross-Encoder Re-ranker ]
                                      │
                                      ▼
                           [ Resolved Case Retrieval ]

1. **Text Normalization**: Custom preprocessing that unifies orthography for MSA and accommodates colloquial dialects.
2. **Topic Categorization**: Classification into key verticals (`billing`, `outage`, `permits`, `account_access`, etc.) with XLM-R.
3. **Dense Retrieval**: Embedding generation with normalized vectors coupled with an inner product **FAISS** index for fast, sub-millisecond retrieval.
4. **Context Re-ranking**: Cross-encoder validation to re-score the top 10 retrieved candidates for accurate alignment.
# Results & Evaluation :
### 1. Topic Classifier Performance
- **Global Macro-F1**: `0.814`
- **95% Bootstrap Confidence Interval**: `[0.798, 0.827]` (extremely narrow, indicating highly stable model parameters)

### 2. Segmented Slices Analysis (Slices F1-Score)
- **English Feedback**: `0.828` 
- **Arabic Feedback**: `0.797` 
- **Weakest Slices**:
  - **Long Messages (Word Count > 25)**: `0.694` (mainly due to topic overlap and ambiguity)
  - **Egyptian Arabic Dialect**: `0.700` (due to spelling variations and lower representation in training data compared to Gulf dialects)

### 3. Behavioral Testing (MFT & Invariance)
- **Minimum Functionality Test (MFT)**: `100.0%` (passed standard unit test classifications for billing, outage, and account access)
- **Invariance Rate (City Name Permutation)**: `100.0%` (the model remains invariant when shifting entities from Dammam to Jeddah or Riyadh)

### 4. Semantic Search Metrics
- **Recall@10**: `0.308` | **MRR@10**: `0.215`
- **Error Diagnosis**: Analysis of failure cases showed that while exact target `case_id`s were occasionally missed, the **correct topic was present in the top 10 in 83% (75/90) of failed queries**. This highlights the semantic robustness of the search index despite template noise.
# Technologies Used :
- **Languages**: Python 3
- **Deep Learning**: PyTorch, Hugging Face Transformers (`pipeline`, `AutoModel`)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **NLP Tools**: CAMeL Tools (for Arabic clitics and morphological segmentation)
- **Evaluation**: Scikit-Learn, Pandas, NumPy
# How to Run the Project (make serve)
# Future Improvements: 
1. Data Augmentation for Weak Slices: Collect and annotate more Egyptian and Levantine dialectal feedback to shrink the performance gap.
2. Multi-Label Classification: Upgrade the classifier pipeline to support multi-label predictions for longer feedback inputs that express more than one complaint.
3. Relevance Alignment: Train a custom Bi-Encoder on domain-specific query-to-resolution pairs to boost the current Recall@10 above the target 0.80 threshold.

# SDAIA Academy GitHub Repository Link:
https://github.com/SDAIAAcademy
