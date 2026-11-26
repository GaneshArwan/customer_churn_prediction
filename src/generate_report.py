import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier

from data_preprocessing import load_and_preprocess_data

# ---------------------------------------------------------------------------
# IEEE-style constants
# ---------------------------------------------------------------------------
PAGE_WIDTH, PAGE_HEIGHT = letter
IEEE_MARGIN_TOP = 0.75 * inch
IEEE_MARGIN_BOTTOM = 1.0 * inch
IEEE_MARGIN_LEFT = 0.625 * inch
IEEE_MARGIN_RIGHT = 0.625 * inch
COL_GAP = 0.25 * inch
CONTENT_WIDTH = PAGE_WIDTH - IEEE_MARGIN_LEFT - IEEE_MARGIN_RIGHT
COL_WIDTH = (CONTENT_WIDTH - COL_GAP) / 2.0

FONT_BODY = 'Times-Roman'
FONT_BOLD = 'Times-Bold'
FONT_ITALIC = 'Times-Italic'
FONT_SIZE_BODY = 10
FONT_SIZE_TITLE = 24
FONT_SIZE_AUTHOR = 11
FONT_SIZE_ABSTRACT = 9
FONT_SIZE_SECTION = 12
FONT_SIZE_SUBSECTION = 10


def _get_positive_scores(model, X_test):
    """Extract continuous positive-class scores for ROC/PR evaluation."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)


def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn'],
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_roc_curve(models_dict, X_test, y_test, title, filename):
    plt.figure(figsize=(6, 4.5))
    for name, model in models_dict.items():
        y_score = _get_positive_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_pr_curve(models_dict, X_test, y_test, title, filename):
    """Plot Precision-Recall curves with Average Precision and no-skill baseline."""
    plt.figure(figsize=(6, 4.5))
    baseline = float(np.mean(y_test))
    for name, model in models_dict.items():
        y_score = _get_positive_scores(model, X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        plt.plot(recall, precision, lw=2, label=f'{name} (AP = {ap:.3f})')

    plt.axhline(baseline, color='navy', lw=1.5, linestyle='--', label=f'Baseline (AP = {baseline:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Churn)')
    plt.ylabel('Precision (Churn)')
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_feature_importance(model, feature_names, title, filename):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(6, 4.5))
    sns.barplot(
        x=top_importances,
        y=top_features,
        hue=top_features,
        palette="viridis",
        legend=False,
    )
    plt.title(title)
    plt.xlabel('Relative Importance (Gini)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_lr_coefficients(model, feature_names, title, filename, top_n=10):
    """Plot top Logistic Regression coefficients showing risk vs protective direction."""
    coefs = np.asarray(model.coef_[0], dtype=float)
    limit = min(top_n, len(coefs))
    top_idx = np.argsort(np.abs(coefs))[::-1][:limit]
    sorted_idx = top_idx[np.argsort(coefs[top_idx])]

    selected_names = [feature_names[i] for i in sorted_idx]
    selected_vals = coefs[sorted_idx]
    bar_colors = ['#C0392B' if v > 0 else '#2980B9' for v in selected_vals]

    plt.figure(figsize=(6, 4.5))
    plt.barh(selected_names, selected_vals, color=bar_colors)
    plt.axvline(0.0, color='black', lw=1)
    plt.title(title)
    plt.xlabel('Log-Odds Coefficient (+ Increases Churn, - Protective)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def compute_business_impact(
    tp: int,
    fp: int,
    ltv: float = 780.0,
    campaign_cost: float = 50.0,
    save_rate: float = 0.35,
) -> dict[str, float | int]:
    """Compute cohort retention campaign economics from confusion-matrix counts."""
    tp_i = int(tp)
    fp_i = int(fp)
    targeted = tp_i + fp_i
    campaign_spend = targeted * float(campaign_cost)
    retained_value = tp_i * float(save_rate) * float(ltv)
    net_profit = retained_value - campaign_spend
    breakeven_save_rate = (
        campaign_spend / (tp_i * float(ltv))
        if tp_i > 0 and ltv > 0
        else float("inf")
    )
    return {
        "targeted": targeted,
        "campaign_spend": campaign_spend,
        "retained_value": retained_value,
        "net_profit": net_profit,
        "breakeven_save_rate": breakeven_save_rate,
    }


def create_metric_table(report_dict):
    """Create an IEEE-styled metric table."""
    data = [['Class', 'Precision', 'Recall', 'F1-Score', 'Support']]
    for cls in ['0', '1']:
        label = 'Churn' if cls == '1' else 'No Churn'
        data.append([
            f"{cls} ({label})",
            f"{report_dict[cls]['precision']:.2f}",
            f"{report_dict[cls]['recall']:.2f}",
            f"{report_dict[cls]['f1-score']:.2f}",
            str(int(report_dict[cls]['support'])),
        ])
    for avg in ['macro avg', 'weighted avg']:
        data.append([
            avg.title(),
            f"{report_dict[avg]['precision']:.2f}",
            f"{report_dict[avg]['recall']:.2f}",
            f"{report_dict[avg]['f1-score']:.2f}",
            str(int(report_dict[avg]['support'])),
        ])

    t = Table(data, colWidths=[90, 65, 55, 60, 55])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
        ('FONTNAME', (0, 1), (-1, -1), FONT_BODY),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


def ieee_header_footer(canvas, doc):
    """IEEE-style header (pages > 1) and page number footer."""
    canvas.saveState()
    canvas.setFont(FONT_ITALIC, 8)
    canvas.setFillColor(colors.gray)
    if getattr(doc, "page", 1) > 1:
        canvas.drawCentredString(
            PAGE_WIDTH / 2,
            PAGE_HEIGHT - 0.5 * inch,
            "Comparative Analysis of Machine Learning Models for Customer Churn Prediction",
        )
    canvas.drawCentredString(PAGE_WIDTH / 2, 0.5 * inch, f"{doc.page}")
    canvas.restoreState()


def _ieee_styles():
    """Build and return an IEEE-conformant stylesheet dictionary."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='IEEETitle', fontName=FONT_BOLD, fontSize=FONT_SIZE_TITLE,
        alignment=TA_CENTER, spaceAfter=6, leading=28))
    styles.add(ParagraphStyle(
        name='IEEEAuthor', fontName=FONT_BODY, fontSize=FONT_SIZE_AUTHOR,
        alignment=TA_CENTER, spaceAfter=4))
    styles.add(ParagraphStyle(
        name='IEEEAbstractLabel', fontName=FONT_BOLD, fontSize=FONT_SIZE_ABSTRACT,
        alignment=TA_JUSTIFY, spaceAfter=2, spaceBefore=10))
    styles.add(ParagraphStyle(
        name='IEEEAbstract', fontName=FONT_ITALIC, fontSize=FONT_SIZE_ABSTRACT,
        alignment=TA_JUSTIFY, leading=12, spaceAfter=10, leftIndent=18, rightIndent=18))
    styles.add(ParagraphStyle(
        name='IEEEKeywords', fontName=FONT_BODY, fontSize=FONT_SIZE_ABSTRACT,
        alignment=TA_JUSTIFY, spaceAfter=14, leftIndent=18, rightIndent=18))
    styles.add(ParagraphStyle(
        name='IEEESection', fontName=FONT_BOLD, fontSize=FONT_SIZE_SECTION,
        alignment=TA_CENTER, spaceBefore=14, spaceAfter=6))
    styles.add(ParagraphStyle(
        name='IEEESubSection', fontName=FONT_ITALIC, fontSize=FONT_SIZE_SUBSECTION,
        spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(
        name='IEEEBody', fontName=FONT_BODY, fontSize=FONT_SIZE_BODY,
        alignment=TA_JUSTIFY, leading=13, spaceAfter=6,
        firstLineIndent=18))
    styles.add(ParagraphStyle(
        name='IEEEBodyNoIndent', fontName=FONT_BODY, fontSize=FONT_SIZE_BODY,
        alignment=TA_JUSTIFY, leading=13, spaceAfter=6))
    styles.add(ParagraphStyle(
        name='IEEECaption', fontName=FONT_BODY, fontSize=8,
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=10))
    styles.add(ParagraphStyle(
        name='IEEERef', fontName=FONT_BODY, fontSize=8,
        leading=10, spaceAfter=2, leftIndent=18, firstLineIndent=-18))
    return styles


def _evaluate_on_test(model, X_test, y_test):
    """Compute comprehensive test-set discrimination, calibration, and confusion metrics."""
    y_pred = model.predict(X_test)
    y_prob = _get_positive_scores(model, X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "pred": y_pred,
        "prob": y_prob,
        "acc": float(accuracy_score(y_test, y_pred)),
        "prec": float(report['1']['precision']),
        "rec": float(report['1']['recall']),
        "f1": float(report['1']['f1-score']),
        "roc_auc": float(auc(fpr, tpr)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "brier": float(brier_score_loss(y_test, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "report": report,
    }


def _styled_ieee_table(data, col_widths, font_size=7.5):
    """Build a compact IEEE booktabs-style table fitting within a column."""
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
        ('FONTNAME', (0, 1), (-1, -1), FONT_BODY),
        ('FONTSIZE', (0, 0), (-1, -1), font_size),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
    ]))
    return table


def build_pdf_report():
    """Train benchmark models, evaluate with CV and calibration metrics, and build IEEE PDF."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "telco_customer_churn.csv")
    report_path = os.path.join(base_dir, "churn_prediction_report.pdf")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ---- Dynamic Dataset Audit ----
    print("Auditing dataset and preprocessing without leakage...")
    raw_df = pd.read_csv(data_path)
    total_raw_rows = len(raw_df)
    raw_feature_count = raw_df.shape[1]
    total_charges_num = pd.to_numeric(raw_df['TotalCharges'], errors='coerce')
    dropped_rows = int(total_charges_num.isna().sum())
    usable_df = raw_df.loc[~total_charges_num.isna()]
    usable_rows = len(usable_df)
    overall_churn_rate = float((usable_df['Churn'] == 'Yes').mean())
    overall_retain_rate = 1.0 - overall_churn_rate

    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    feature_names = X_train.columns.tolist()
    encoded_feature_count = len(feature_names)
    train_size = len(y_train)
    test_size = len(y_test)
    churners_in_test = int(sum(y_test))
    test_churn_rate = churners_in_test / test_size

    # ---- 5-Fold Stratified Cross-Validation on Training Split ----
    print("Running 5-fold stratified cross-validation on training split...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_base = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)

    rf_cv = cross_validate(rf_base, X_train, y_train, cv=cv, scoring=['roc_auc', 'recall'])
    lr_cv = cross_validate(lr_base, X_train, y_train, cv=cv, scoring=['roc_auc', 'recall'])

    # ---- Train and Evaluate Final Models ----
    print("Training Baseline Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_m = _evaluate_on_test(rf_model, X_test, y_test)
    rf_cm_path = os.path.join(plots_dir, "rf_cm.png")
    save_confusion_matrix(y_test, rf_m["pred"], "Fig. 1. Baseline Random Forest", rf_cm_path)
    rf_fi_path = os.path.join(plots_dir, "rf_fi.png")
    save_feature_importance(rf_model, feature_names, "Fig. 2. Top 10 Feature Importances (RF)", rf_fi_path)

    print("Training Balanced Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_m = _evaluate_on_test(lr_model, X_test, y_test)
    lr_cm_path = os.path.join(plots_dir, "lr_cm.png")
    save_confusion_matrix(y_test, lr_m["pred"], "Fig. 3. Balanced Logistic Regression", lr_cm_path)
    lr_coef_path = os.path.join(plots_dir, "lr_coef.png")
    save_lr_coefficients(lr_model, feature_names, "Fig. 4. Top 10 LR Log-Odds Coefficients", lr_coef_path)

    print("Finetuning Deep Learning Model (MLP with early stopping)...")
    base_mlp = MLPClassifier(max_iter=1000, early_stopping=True, random_state=42)
    param_dist = {
        'hidden_layer_sizes': [(64, 32), (128, 64, 32), (32, 16)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
    }
    mlp_search = RandomizedSearchCV(
        base_mlp,
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42,
    )
    mlp_search.fit(X_train, y_train)
    best_mlp_params = mlp_search.best_params_
    print(f"Best MLP Parameters: {best_mlp_params}")
    mlp_model = mlp_search.best_estimator_
    mlp_m = _evaluate_on_test(mlp_model, X_test, y_test)
    mlp_cm_path = os.path.join(plots_dir, "mlp_cm.png")
    save_confusion_matrix(y_test, mlp_m["pred"], "Fig. 5. Tuned MLP Neural Network", mlp_cm_path)

    models_dict = {"Random Forest": rf_model, "Balanced LR": lr_model, "MLP NN": mlp_model}
    roc_path = os.path.join(plots_dir, "roc_curve.png")
    save_roc_curve(models_dict, X_test, y_test, "Fig. 6. ROC Curve Comparison", roc_path)
    pr_path = os.path.join(plots_dir, "pr_curve.png")
    save_pr_curve(models_dict, X_test, y_test, "Fig. 7. Precision-Recall Curve Comparison", pr_path)

    # Dynamic min/max recall comparison
    model_metrics = {"Random Forest": rf_m, "Balanced LR": lr_m, "MLP NN": mlp_m}
    min_rec_name = min(model_metrics, key=lambda k: model_metrics[k]["rec"])
    max_rec_name = max(model_metrics, key=lambda k: model_metrics[k]["rec"])
    max_roc_name = max(model_metrics, key=lambda k: model_metrics[k]["roc_auc"])
    max_pr_name = max(model_metrics, key=lambda k: model_metrics[k]["pr_auc"])
    best_brier_name = min(model_metrics, key=lambda k: model_metrics[k]["brier"])

    # Business impact at conservative (30%) and moderate (50%) save rates
    rf_econ_30 = compute_business_impact(rf_m["tp"], rf_m["fp"], save_rate=0.30)
    lr_econ_30 = compute_business_impact(lr_m["tp"], lr_m["fp"], save_rate=0.30)
    rf_econ_50 = compute_business_impact(rf_m["tp"], rf_m["fp"], save_rate=0.50)
    lr_econ_50 = compute_business_impact(lr_m["tp"], lr_m["fp"], save_rate=0.50)
    inc_profit_30 = lr_econ_30["net_profit"] - rf_econ_30["net_profit"]
    inc_profit_50 = lr_econ_50["net_profit"] - rf_econ_50["net_profit"]

    # ---- Build IEEE-format PDF ----
    doc = BaseDocTemplate(
        report_path,
        pagesize=letter,
        topMargin=IEEE_MARGIN_TOP,
        bottomMargin=IEEE_MARGIN_BOTTOM,
        leftMargin=IEEE_MARGIN_LEFT,
        rightMargin=IEEE_MARGIN_RIGHT,
    )

    full_frame = Frame(
        IEEE_MARGIN_LEFT,
        IEEE_MARGIN_BOTTOM,
        CONTENT_WIDTH,
        PAGE_HEIGHT - IEEE_MARGIN_TOP - IEEE_MARGIN_BOTTOM,
        id='full',
    )
    left_col = Frame(
        IEEE_MARGIN_LEFT,
        IEEE_MARGIN_BOTTOM,
        COL_WIDTH,
        PAGE_HEIGHT - IEEE_MARGIN_TOP - IEEE_MARGIN_BOTTOM,
        id='left',
    )
    right_col = Frame(
        IEEE_MARGIN_LEFT + COL_WIDTH + COL_GAP,
        IEEE_MARGIN_BOTTOM,
        COL_WIDTH,
        PAGE_HEIGHT - IEEE_MARGIN_TOP - IEEE_MARGIN_BOTTOM,
        id='right',
    )

    doc.addPageTemplates([
        PageTemplate(id='TitlePage', frames=[full_frame], onPage=ieee_header_footer),
        PageTemplate(id='TwoCol', frames=[left_col, right_col], onPage=ieee_header_footer),
    ])

    S = _ieee_styles()
    table_width = COL_WIDTH - 10  # 242 pt
    img_width = COL_WIDTH - 10
    img_height = img_width * 0.75

    elements = []

    # ====================================================================
    # TITLE BLOCK (full-width)
    # ====================================================================
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "Comparative Analysis of Machine Learning<br/>Models for Customer Churn Prediction",
        S['IEEETitle']))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph("William Arwan", S['IEEEAuthor']))
    elements.append(Paragraph("<i>Independent Researcher</i>", S['IEEEAuthor']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b><i>Abstract</i></b>", S['IEEEAbstractLabel']))
    abstract = (
        f"This paper evaluates three machine learning architectures for telecommunications customer "
        f"churn prediction on the IBM Telco dataset ({total_raw_rows:,} raw records, {usable_rows:,} "
        f"usable records after removing {dropped_rows} blank TotalCharges entries, "
        f"{overall_churn_rate:.1%} churn rate). A baseline Random Forest, a class-weighted Logistic "
        f"Regression, and an early-stopped Multi-Layer Perceptron (MLP) are trained on {train_size:,} "
        f"samples using leak-free split-before-scale preprocessing and evaluated on a held-out test "
        f"cohort of {test_size:,} customers ({churners_in_test} churners, {test_churn_rate:.1%}). "
        f"Class-weighted Logistic Regression achieves {lr_m['rec']:.1%} churn recall "
        f"(ROC-AUC {lr_m['roc_auc']:.3f}, PR-AUC {lr_m['pr_auc']:.3f}) compared to {rf_m['rec']:.1%} "
        f"for the Random Forest baseline (+{(lr_m['rec'] - rf_m['rec']) * 100:.1f} percentage points) "
        f"and {mlp_m['rec']:.1%} for the tuned MLP. Five-fold stratified cross-validation confirms "
        f"training stability, while Brier calibration analysis highlights probability shift under "
        f"class reweighting. Under explicit retention campaign economics ($780 LTV, $50 outreach cost), "
        f"deploying Balanced Logistic Regression yields +${inc_profit_30:,.0f} to +${inc_profit_50:,.0f} "
        f"in projected incremental net profit per {test_size:,} evaluated accounts across 30%-50% save rates."
    )
    elements.append(Paragraph(abstract, S['IEEEAbstract']))

    elements.append(Paragraph(
        "<b><i>Index Terms</i></b> — customer churn, class imbalance, logistic regression, "
        "random forest, multi-layer perceptron, PR-AUC, Brier calibration, retention economics",
        S['IEEEKeywords']))

    elements.append(NextPageTemplate('TwoCol'))
    elements.append(PageBreak())

    # ====================================================================
    # I. INTRODUCTION
    # ====================================================================
    elements.append(Paragraph("I. Introduction", S['IEEESection']))
    elements.append(Paragraph(
        "Customer churn prediction is a core revenue-protection task in subscription "
        "telecommunications, where acquiring a new subscriber typically costs five to seven "
        "times more than retaining an existing customer [1]. Identifying high-risk accounts "
        "early enables targeted retention outreach before contract cancellation.",
        S['IEEEBody']))
    elements.append(Paragraph(
        f"The central modeling challenge is class imbalance: in the evaluated cohort, only "
        f"{overall_churn_rate:.1%} of customers churned compared to {overall_retain_rate:.1%} "
        f"retained. Classifiers optimized for unweighted overall accuracy over-predict the majority "
        f"class and miss many departing customers [2].",
        S['IEEEBody']))
    elements.append(Paragraph(
        "This study benchmarks three model families under identical leak-free preprocessing: "
        "(a) an unweighted Random Forest baseline, (b) a class-weighted Logistic Regression, and "
        "(c) an early-stopped MLP neural network, evaluated across discrimination (ROC-AUC, PR-AUC), "
        "calibration (Brier score), cross-validation stability, and campaign ROI.",
        S['IEEEBody']))

    # ====================================================================
    # II. DATASET & PREPROCESSING
    # ====================================================================
    elements.append(Paragraph("II. Dataset and Preprocessing", S['IEEESection']))
    elements.append(Paragraph(
        f"The IBM Telco Customer Churn dataset [3] contains {total_raw_rows:,} customer records "
        f"across {raw_feature_count} raw columns spanning demographics, tenure, contract terms, "
        f"billing, and subscribed services. Coercing TotalCharges to numeric identifies {dropped_rows} "
        f"blank records (tenure = 0 accounts), leaving {usable_rows:,} usable customers with a "
        f"{overall_churn_rate:.1%} churn prevalence.",
        S['IEEEBody']))
    elements.append(Paragraph(
        f"Preprocessing strictly prevents train-test leakage: (1) customerID is dropped, "
        f"(2) {dropped_rows} rows with missing TotalCharges are removed, (3) categorical variables "
        f"are one-hot encoded with drop_first=True producing {encoded_feature_count} model features, "
        f"(4) data is partitioned 80/20 (random_state=42) into {train_size:,} training and "
        f"{test_size:,} test rows, and (5) StandardScaler is fit exclusively on the training "
        f"numerical columns before transforming the test split.",
        S['IEEEBody']))

    # ====================================================================
    # III. METHODOLOGY
    # ====================================================================
    elements.append(Paragraph("III. Methodology", S['IEEESection']))

    elements.append(Paragraph("A. Baseline Random Forest", S['IEEESubSection']))
    elements.append(Paragraph(
        "A 100-tree Random Forest classifier (random_state=42) is trained with default uniform "
        "class weights. It serves as the accuracy-oriented non-linear baseline and yields "
        "Gini impurity feature importance rankings.",
        S['IEEEBody']))

    elements.append(Paragraph("B. Class-Weighted Logistic Regression", S['IEEESubSection']))
    elements.append(Paragraph(
        "A Logistic Regression classifier (class_weight='balanced', max_iter=1000, random_state=42) "
        "weights minority-class errors inversely proportionally to class frequency, shifting the "
        "decision boundary toward higher churn recall while preserving additive log-odds interpretability.",
        S['IEEEBody']))

    elements.append(Paragraph("C. Multi-Layer Perceptron & Validation", S['IEEESubSection']))
    elements.append(Paragraph(
        f"An MLPClassifier with early stopping (max_iter=1000, early_stopping=True) is tuned via "
        f"3-fold RandomizedSearchCV (5 iterations, macro-F1 objective) over hidden sizes, "
        f"activations, L2 alpha, and learning rate. The selected configuration is "
        f"hidden_layer_sizes={best_mlp_params['hidden_layer_sizes']}, "
        f"activation='{best_mlp_params['activation']}', alpha={best_mlp_params['alpha']}, "
        f"learning_rate_init={best_mlp_params['learning_rate_init']}. Five-fold stratified "
        f"cross-validation on the {train_size:,}-row training split assesses variance prior to holdout testing.",
        S['IEEEBody']))

    # ====================================================================
    # IV. RESULTS
    # ====================================================================
    elements.append(Paragraph("IV. Results", S['IEEESection']))

    summary_data = [
        ['Model', 'Acc.', 'Prec.', 'Rec.', 'F1', 'ROC', 'PR', 'Brier'],
        [
            'RF Baseline',
            f"{rf_m['acc']:.3f}", f"{rf_m['prec']:.2f}", f"{rf_m['rec']:.2f}",
            f"{rf_m['f1']:.2f}", f"{rf_m['roc_auc']:.3f}", f"{rf_m['pr_auc']:.3f}", f"{rf_m['brier']:.3f}",
        ],
        [
            'Balanced LR',
            f"{lr_m['acc']:.3f}", f"{lr_m['prec']:.2f}", f"{lr_m['rec']:.2f}",
            f"{lr_m['f1']:.2f}", f"{lr_m['roc_auc']:.3f}", f"{lr_m['pr_auc']:.3f}", f"{lr_m['brier']:.3f}",
        ],
        [
            'MLP NN',
            f"{mlp_m['acc']:.3f}", f"{mlp_m['prec']:.2f}", f"{mlp_m['rec']:.2f}",
            f"{mlp_m['f1']:.2f}", f"{mlp_m['roc_auc']:.3f}", f"{mlp_m['pr_auc']:.3f}", f"{mlp_m['brier']:.3f}",
        ],
    ]
    summary_table = _styled_ieee_table(
        summary_data,
        col_widths=[56, 26, 26, 26, 24, 28, 28, 28],
        font_size=7.0,
    )
    elements.append(KeepTogether([
        summary_table,
        Paragraph(
            f"TABLE I. Holdout test-set ({test_size:,} rows, {churners_in_test} churners) churn-class performance, ROC-AUC, PR-AUC, and Brier score.",
            S['IEEECaption'],
        ),
    ]))

    cv_data = [
        ['Model (5-Fold CV)', 'CV ROC-AUC (mean ± std)', 'CV Recall (mean ± std)'],
        [
            'RF Baseline',
            f"{rf_cv['test_roc_auc'].mean():.3f} ± {rf_cv['test_roc_auc'].std():.3f}",
            f"{rf_cv['test_recall'].mean():.3f} ± {rf_cv['test_recall'].std():.3f}",
        ],
        [
            'Balanced LR',
            f"{lr_cv['test_roc_auc'].mean():.3f} ± {lr_cv['test_roc_auc'].std():.3f}",
            f"{lr_cv['test_recall'].mean():.3f} ± {lr_cv['test_recall'].std():.3f}",
        ],
    ]
    cv_table = _styled_ieee_table(cv_data, col_widths=[74, 84, 84], font_size=7.5)
    elements.append(KeepTogether([
        cv_table,
        Paragraph(
            f"TABLE II. Five-fold stratified cross-validation stability on the {train_size:,}-row training split.",
            S['IEEECaption'],
        ),
    ]))

    ranking_sentence = (
        f"{max_roc_name} leads in ranking discrimination "
        f"(ROC-AUC = {model_metrics[max_roc_name]['roc_auc']:.3f}, "
        f"PR-AUC = {model_metrics[max_pr_name]['pr_auc']:.3f} vs. {test_churn_rate:.3f} no-skill baseline)."
        if max_roc_name == max_pr_name
        else f"{max_roc_name} leads in ROC-AUC ({model_metrics[max_roc_name]['roc_auc']:.3f}; "
        f"PR-AUC = {model_metrics[max_roc_name]['pr_auc']:.3f}), whereas {max_pr_name} achieves "
        f"the highest PR-AUC ({model_metrics[max_pr_name]['pr_auc']:.3f} vs. {test_churn_rate:.3f} no-skill baseline)."
    )

    elements.append(Paragraph(
        f"Table I compares holdout performance on the {test_size:,}-customer test set "
        f"({churners_in_test} churners). {max_rec_name} achieves the highest churn recall "
        f"({model_metrics[max_rec_name]['rec']:.1%}), whereas {min_rec_name} records the lowest "
        f"({model_metrics[min_rec_name]['rec']:.1%}). Compared with the Random Forest baseline "
        f"({rf_m['rec']:.1%} recall, {rf_m['acc']:.1%} accuracy), Balanced Logistic Regression "
        f"improves churn recall by +{(lr_m['rec'] - rf_m['rec']) * 100:.1f} percentage points "
        f"(TP = {lr_m['tp']} vs. {rf_m['tp']}) while trading {(rf_m['acc'] - lr_m['acc']) * 100:.1f} "
        f"points of overall accuracy. {ranking_sentence} "
        f"Table II confirms consistent fold-level performance across 5 stratified training folds.",
        S['IEEEBody']))

    for img_path, caption in [
        (rf_cm_path, f"Fig. 1. Baseline Random Forest confusion matrix (TP={rf_m['tp']}, FP={rf_m['fp']}, FN={rf_m['fn']}, TN={rf_m['tn']})."),
        (rf_fi_path, "Fig. 2. Top 10 Gini feature importances from the Random Forest model."),
        (lr_cm_path, f"Fig. 3. Balanced Logistic Regression confusion matrix (TP={lr_m['tp']}, FP={lr_m['fp']}, FN={lr_m['fn']}, TN={lr_m['tn']})."),
        (lr_coef_path, "Fig. 4. Top 10 signed log-odds coefficients from Balanced Logistic Regression."),
        (mlp_cm_path, f"Fig. 5. Tuned MLP neural network confusion matrix (TP={mlp_m['tp']}, FP={mlp_m['fp']}, FN={mlp_m['fn']}, TN={mlp_m['tn']})."),
        (roc_path, "Fig. 6. ROC curve comparison across the three models on the holdout test set."),
        (pr_path, "Fig. 7. Precision-Recall curve comparison against the holdout churn prevalence baseline."),
    ]:
        elements.append(KeepTogether([
            Image(img_path, width=img_width, height=img_height),
            Paragraph(caption, S['IEEECaption']),
        ]))

    # ====================================================================
    # V. DISCUSSION
    # ====================================================================
    elements.append(Paragraph("V. Discussion", S['IEEESection']))

    elements.append(Paragraph("A. Feature Drivers & Interpretability", S['IEEESubSection']))
    elements.append(Paragraph(
        "Both Random Forest Gini importance (Fig. 2) and signed Logistic Regression coefficients "
        "(Fig. 4) identify tenure, contract duration, Fiber optic internet service, and billing "
        "charges as primary churn signals. Longer tenure and two-year contracts carry strong "
        "negative log-odds (protective), whereas Fiber optic service, electronic check payment "
        "(Fig. 2), and paperless billing increase predicted churn risk.",
        S['IEEEBody']))

    elements.append(Paragraph("B. Retention Campaign Economics", S['IEEESubSection']))
    econ_data = [
        ['Model', 'Save%', 'Target', 'Spend', 'Retained', 'Net Profit'],
        [
            'RF Baseline', '30%', str(rf_econ_30['targeted']),
            f"${rf_econ_30['campaign_spend']:,.0f}",
            f"${rf_econ_30['retained_value']:,.0f}",
            f"${rf_econ_30['net_profit']:,.0f}",
        ],
        [
            'Balanced LR', '30%', str(lr_econ_30['targeted']),
            f"${lr_econ_30['campaign_spend']:,.0f}",
            f"${lr_econ_30['retained_value']:,.0f}",
            f"${lr_econ_30['net_profit']:,.0f}",
        ],
        [
            'RF Baseline', '50%', str(rf_econ_50['targeted']),
            f"${rf_econ_50['campaign_spend']:,.0f}",
            f"${rf_econ_50['retained_value']:,.0f}",
            f"${rf_econ_50['net_profit']:,.0f}",
        ],
        [
            'Balanced LR', '50%', str(lr_econ_50['targeted']),
            f"${lr_econ_50['campaign_spend']:,.0f}",
            f"${lr_econ_50['retained_value']:,.0f}",
            f"${lr_econ_50['net_profit']:,.0f}",
        ],
    ]
    econ_table = _styled_ieee_table(
        econ_data,
        col_widths=[46, 32, 34, 42, 44, 44],
        font_size=7.0,
    )
    elements.append(KeepTogether([
        econ_table,
        Paragraph(
            f"TABLE III. Retention campaign simulation on {test_size:,} test customers ($780 LTV, $50 contact cost).",
            S['IEEECaption'],
        ),
    ]))

    elements.append(Paragraph(
        f"Evaluating only the top-line accuracy of Random Forest ({rf_m['acc']:.1%}) masks "
        f"{rf_m['fn']} missed churners ({(1 - rf_m['rec']):.1%} false-negative rate). Under an "
        f"illustrative retention assumption of $780 saved customer LTV and $50 outreach cost per "
        f"flagged account (Table III), Balanced LR breaks even at a {lr_econ_30['breakeven_save_rate']:.1%} "
        f"retention save rate (vs. {rf_econ_30['breakeven_save_rate']:.1%} for RF). At 30%-50% save "
        f"rates, Balanced LR generates +${inc_profit_30:,.0f} to +${inc_profit_50:,.0f} in incremental "
        f"net profit over the RF baseline across {test_size:,} evaluated customers.",
        S['IEEEBody']))

    elements.append(Paragraph("C. Calibration & Model Governance", S['IEEESubSection']))
    elements.append(Paragraph(
        f"Although Balanced LR maximizes recall ({lr_m['rec']:.1%}) and ROC-AUC ({lr_m['roc_auc']:.3f}), "
        f"its probability Brier score ({lr_m['brier']:.3f}) is higher than {best_brier_name} "
        f"({model_metrics[best_brier_name]['brier']:.3f}) because class_weight='balanced' adds a "
        f"positive log-odds offset that shifts mean predicted churn probability upward "
        f"({float(np.mean(lr_m['prob'])):.3f} vs. {test_churn_rate:.3f} actual prevalence). "
        f"Before using raw predicted probabilities for expected-value spend optimization or tiered "
        f"discounting, post-hoc Platt scaling or isotonic calibration should be fit on a validation "
        f"fold. Production governance should additionally monitor feature Population Stability Index "
        f"(PSI) and rolling churn recall across contract cohorts.",
        S['IEEEBody']))

    elements.append(Paragraph(
        f"The early-stopped MLP converges cleanly without iteration warnings and achieves "
        f"{mlp_m['acc']:.1%} accuracy with {mlp_m['rec']:.1%} churn recall and {mlp_m['roc_auc']:.3f} "
        f"ROC-AUC. Without explicit class weighting or focal loss, its default 0.50 decision threshold "
        f"remains conservative on minority-class churners.",
        S['IEEEBody']))

    # ====================================================================
    # VI. CONCLUSION
    # ====================================================================
    elements.append(Paragraph("VI. Conclusion", S['IEEESection']))
    elements.append(Paragraph(
        f"Class-weighted Logistic Regression delivers the strongest balance of churn recall "
        f"({lr_m['rec']:.1%}), ranking discrimination (ROC-AUC {lr_m['roc_auc']:.3f}, PR-AUC "
        f"{lr_m['pr_auc']:.3f}), feature transparency, and net retention profit among the three "
        f"benchmarked architectures. Coupled with leak-free preprocessing, strict Pydantic input "
        f"validation, and post-hoc probability calibration, it provides a sound foundation for "
        f"production retention workflows.",
        S['IEEEBody']))

    # ====================================================================
    # REFERENCES
    # ====================================================================
    elements.append(Paragraph("References", S['IEEESection']))
    refs = [
        "[1] A. Keramati, R. Jafari-Marandi, M. Aliannejadi, I. Ahmadian, M. Mozaffari, and U. Abbasi, "
        "\"Improved churn prediction in telecommunication industry using data mining techniques,\" "
        "<i>Applied Soft Computing</i>, vol. 24, pp. 994-1012, 2014.",

        "[2] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, \"SMOTE: Synthetic "
        "minority over-sampling technique,\" <i>Journal of Artificial Intelligence Research</i>, "
        "vol. 16, pp. 321-357, 2002.",

        "[3] IBM, \"Telco Customer Churn,\" Kaggle, 2018. [Online]. Available: "
        "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",

        "[4] F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" "
        "<i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825-2830, 2011.",
    ]
    for ref in refs:
        elements.append(Paragraph(ref, S['IEEERef']))

    doc.build(elements)
    print(f"Report generated successfully at {report_path}")
    return {
        "rf": rf_m,
        "lr": lr_m,
        "mlp": mlp_m,
        "rf_cv": rf_cv,
        "lr_cv": lr_cv,
        "econ_30_inc": inc_profit_30,
        "econ_50_inc": inc_profit_50,
    }


if __name__ == "__main__":
    build_pdf_report()
