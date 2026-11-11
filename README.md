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
# Workflow / Architecture
# Results & Evaluation 
# Technologies Used
# How to Run the Project (make serve)
# Future Improvements
# SDAIA Academy GitHub Repository Link
