# Careery Portfolio Guide

## Portfolio Presentation Essentials
- **GitHub structure matters.** Every project gets its own repository with a clean README, a requirements.txt, and organized folders (data/, notebooks/, src/).
- **Deploy at least one model.** A Streamlit or Gradio app on Streamlit Cloud or Hugging Face Spaces turns a static notebook into an interactive demo a recruiter can click.
- **Write one technical blog post.** Explaining methodology in plain language - why a particular model was chosen, what trade-offs were made - proves communication skills that notebooks alone cannot.
- **GitHub profile README.** Create a profile-level README that links to all portfolio projects with one-line descriptions and links to live demos.

## Resume Integration
Every portfolio project should translate into a resume bullet with quantified impact. Use the formula: `[tool/technique] + [model metric] + [business result]`. 
For example: *"Built a churn prediction model using XGBoost (AUC 0.91) that identified 2,300 at-risk accounts, enabling a retention campaign that saved $1.2M ARR."*

## What Makes a Great Data Science Portfolio Project
**Portfolio Mistakes That Get Projects Ignored:**
- **Using only Titanic, Iris, or MNIST datasets:** Hiring managers see these dozens of times per hiring cycle - they signal tutorial completion, not data science ability. Use real-world datasets from UCI ML Repository, government open data, or web scraping projects.
- **Skipping feature engineering and jumping straight to modeling:** In production data science, feature engineering drives 80% of model performance - skipping it looks amateur. Include a dedicated feature engineering section showing domain-informed transformations.
- **Reporting only accuracy with no model evaluation depth:** Accuracy on imbalanced data is meaningless - hiring managers look for precision, recall, F1, AUC, and confusion matrices. Include multiple evaluation metrics, cross-validation results, and honest discussion of model limitations.
- **No README or business framing:** A Jupyter notebook without context is meaningless to someone who doesn't know the dataset or the problem. Every project gets a README with: problem statement, data source, methodology, results, and business implications.

**Key Takeaway:**
Intermediate projects demonstrate the ability to apply specialized ML techniques - NLP, time series, recommendation systems, statistical experimentation, and deep learning - to realistic problems. These projects carry the most weight in hiring evaluations because they're closest to actual production data science work.
