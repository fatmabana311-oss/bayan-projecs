# Final-Project--Bayan-bilingual-citizen-feedback
# Prepared by : Shahad altulyan , fatmah aljeaithen 
[https://github.com/shahad030/Final-Project--Bayan-bilingual-citizen-feedback/blob/main/README.md](https://github.com/fatmabana311-oss)
# Project Overview: 
Bayan turns Arabic/English citizen messages into structured cases: topic classification, sentiment, entity extraction, and similar-case retrieval. Built by integrating the components developed in Labs 1–7 (SDAIA Academy, SDA-AIE-211).
This is an AI-powered system designed to process citizen complaints and suggestions. It serves municipal operations by automatically sorting incoming text requests into administrative categories and allowing agents to immediately find historically resolved cases that match new incoming issues—even if they are written in a different language or dialect.
# Problem Description: 
Citizens send free-text feedback (Arabic and English, mixed dialects, emoji, phone numbers, elongation). Raw messages must be cleaned safely (PII masked first) and converted into: a classified topic, extracted entities, and a list of similar past cases.
# Dataset & Model Used:
- **Dataset**: 
  - **`bayan_feedback_clean.csv`**: Over 3,400 cleaned, labeled, and annotated feedback items.
  - **`bayan_cases.csv`**: A repository of 20,000 resolved historical cases used for building the semantic search database.
  - **`bayan_queries.jsonl`**: Standardized, human-judged test queries used to evaluate search metrics.
- **Models Used**:
 - **Data:** `bayan_feedback.csv` — 12,000 messages (6,000 Arabic, 6,000 English), columns include topic, sentiment, dialect_region. Cleaned version: `bayan_feedback_clean.csv` (output of Lab 1).
- **Preprocessing:** `src/bayan/preprocess.py`, version `bayan-preproc-v1` (mask PII → Unicode NFC → remove tatweel → normalize whitespace).
- **Tokenizer audit (Lab 1, measured fertility):**

| Model | ar | en |
|---|---|---|
| XLM-R | 1.60 | 1.15 |
| mBERT | 2.23 | 1.15 |
| CAMeLBERT | 1.29 | 2.45 |

  Decision: Arabic → CAMeLBERT; bilingual → XLM-R. [TODO: confirm which checkpoint `bayan_topic_model` is actually based on]
- **Transformer anatomy (Lab 2, measured):** our attention and multi-head attention implementations matched PyTorch's outputs (`allclose` = True); masks verified on real Bayan sentences; the padding regression test passed with the mask (True) and failed without it (False). With the `xlm-roberta-base` tokenizer, p95 token length is 11 overall (Arabic 13, English 10); `MAX_LENGTH = 11` truncates 4.2% of messages.
- **Classifier:** `bayan_topic_model` (Lab 3a). [TODO: base model, number of classes]
- **NER (Lab 3b):** `xlm-roberta-base` fine-tuned for token classification on `bayan_ner_4000.csv` (3,400 train / 600 validation; tags PERSON, LOCATION, SERVICE), 1 epoch, scored with entity-level F1 (seqeval). Saved as `bayan_ner_model`.
- **QA (Lab 3b):** `xlm-roberta-base` fine-tuned for extractive QA on `bayan_qa_600.jsonl` (90 of 600 questions unanswerable), returns `None` when the null score beats the best span. Saved as `bayan_qa_model`.
- **Search (Day 3):** `paraphrase-multilingual-MiniLM-L12-v2` embeddings + FAISS `IndexFlatIP` (normalized vectors).
# Workflow / Architecture:
 raw message → preprocess (PII mask + normalize) → classifier (ONNX fp32)
                                               → NER
                                               → embeddings → FAISS → re-rank → similar cases


1. **Text Normalization**: Custom preprocessing that unifies orthography for MSA and accommodates colloquial dialects.
2. **Topic Categorization**: Classification into key verticals (`billing`, `outage`, `permits`, `account_access`, etc.) with XLM-R.
3. **Dense Retrieval**: Embedding generation with normalized vectors coupled with an inner product **FAISS** index for fast, sub-millisecond retrieval.
4. **Context Re-ranking**: Cross-encoder validation to re-score the top 10 retrieved candidates for accurate alignment.
# Results & Evaluation :
### 1. xlm-roberta-base Classifier Performance :
Macro F1 : 0.884924	
Accuracy:0.879589
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

# Future Improvements: 
1. Data Augmentation for Weak Slices: Collect and annotate more Egyptian and Levantine dialectal feedback to shrink the performance gap.
2. Multi-Label Classification: Upgrade the classifier pipeline to support multi-label predictions for longer feedback inputs that express more than one complaint.
3. Relevance Alignment: Train a custom Bi-Encoder on domain-specific query-to-resolution pairs to boost the current Recall@10 above the target 0.80 threshold.

# SDAIA Academy GitHub Repository Link:
https://github.com/SDAIAAcademy
