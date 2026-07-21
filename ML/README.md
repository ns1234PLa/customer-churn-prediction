# Customer Churn Prediction — Machine Learning Pipeline

This repository workspace houses the core predictive engine and data processing pipelines designed to identify high-risk customer accounts. The architecture transitions raw behavioral logs and unstructured customer text into an optimized ensemble modeling pipeline, utilizing advanced Explainable AI (XAI) frameworks to isolate localized churn drivers.

---

##  Pipeline Architecture & Implementation

The machine learning framework is built as a deterministic, decoupled two-stage data engine:

### 1. Feature Synthesis & NLP Integration (`notebook/data_preprocess.ipynb`)
- **Operational Engineering:** Ingests demographic variables and behavioral markers, structuring features around contract longevity, payment vectors, service utilization matrices, and support ticket frequencies.
- **Text Sentiment Extraction:** Implements a text processing sequence that parses qualitative customer support notes and unstructured feedback logs to compute numerical sentiment polarity coefficients. This turns loose textual complaints into highly predictive operational metrics.

### 2. Ensemble Optimization & Explainability (`notebook/generate_full_predictions.ipynb`)
- **Algorithmic Benchmarking:** Evaluates comparative performance metrics across baseline models, including Logistic Regression, Support Vector Machines (SVM), and Random Forests.
- **Imbalance Handling:** Utilizes targeted sample scaling within an **XGBoost Classifier** configuration to effectively mitigate heavy class imbalances typical of real-world corporate churn data without compromising precision-recall curves.
- **Local Feature Attribution:** Deploys a post-training `shap.TreeExplainer` sequence across all individual profiles to compute structural contribution matrices for every feature.

---

##  Pipeline Deliverables & Target Artifacts

Upon execution, the processing pipeline handles data math locally and serializes specific, high-value data structures:

* **`models/best_xgb_model.pkl`** — The serialized binary weights of the optimized tree ensemble, ready for rapid deployment or batch inference scheduling.
* **`metrics/churn_probabilities.csv`** — The final scoring layer mapping unique account identifiers directly to localized risk vector percentages.
* **`metrics/shap_all_customers.csv` (and `.json`)** — A dense mathematical attribution matrix mapping the individual directional force ($+ \text{ or } -$) and magnitude of every single variable per customer record.
* **`plots/shap_summary_plot.png`** — Automated global diagnostic visualizations mapping overall feature impact and structural model stability across the entire sample base.

---

##  Engineering Core: Business Impact Solved

Traditional predictive engines operate as opaque "black boxes," flagging *who* might churn without explaining *why*. This architecture directly solves three primary enterprise bottlenecks:

1. **Captures the Qualitative Voice:** By blending NLP sentiment scores natively with rigid transaction tracking, the model identifies friction points and flags churning behavior from text complaints before the user shifts their bill or service configurations.
2. **Eliminates Opaque Predictions:** Exporting individualized SHAP metrics alongside raw statistical probabilities translates complex tree decisions into transparent operational context. Business teams see exactly *why* an account is at risk (e.g., separating a drop driven by high billing anomalies from a drop driven by technical support failures).
3. **Enables Targeted Retentive Interventions:** Shifting predictions from abstract scores into individualized driver matrices allows account managers to deliver highly targeted, cost-effective customer retention strategies rather than issuing generic promotions blindly.

---

##  Pipeline Execution

To initialize the background engine locally and dynamically regenerate all serialized data models and inference metrics, execute the following commands within this directory directory:

```bash
# Install the core analytics and modeling frameworks
pip install -r requirements.txt

# Execute the pre-processing and model training pipelines sequentially via terminal
jupyter nbconvert --to notebook --execute notebook/data_preprocess.ipynb
jupyter nbconvert --to notebook --execute notebook/generate_full_predictions.ipynb