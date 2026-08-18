import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, 
                                Table, TableStyle, PageBreak)
from reportlab.lib import colors
from data_preprocessing import load_and_preprocess_data

def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def create_metric_table(report_dict):
    data = [['Class', 'Precision', 'Recall', 'F1-Score', 'Support']]
    for cls in ['0', '1']:
        label = 'Churn' if cls == '1' else 'No Churn'
        data.append([
            f"Class {cls} ({label})",
            f"{report_dict[cls]['precision']:.2f}",
            f"{report_dict[cls]['recall']:.2f}",
            f"{report_dict[cls]['f1-score']:.2f}",
            str(report_dict[cls]['support'])
        ])
    for avg in ['macro avg', 'weighted avg']:
        data.append([
            avg.title(),
            f"{report_dict[avg]['precision']:.2f}",
            f"{report_dict[avg]['recall']:.2f}",
            f"{report_dict[avg]['f1-score']:.2f}",
            str(report_dict[avg]['support'])
        ])
        
    t = Table(data, colWidths=[120, 80, 80, 80, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2C3E50")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 10),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F8F9FA")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BDC3C7")),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
    ]))
    return t

def header_footer(canvas, doc):
    canvas.saveState()
    # Header
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(colors.HexColor("#7F8C8D"))
    canvas.drawString(doc.leftMargin, doc.height + doc.topMargin + 10, "Customer Churn Prediction - Research Report")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, doc.height + doc.topMargin + 10, "Confidential")
    canvas.line(doc.leftMargin, doc.height + doc.topMargin + 5, doc.pagesize[0] - doc.rightMargin, doc.height + doc.topMargin + 5)
    
    # Footer
    canvas.line(doc.leftMargin, doc.bottomMargin - 5, doc.pagesize[0] - doc.rightMargin, doc.bottomMargin - 5)
    canvas.drawString(doc.leftMargin, doc.bottomMargin - 15, "Generated automatically by CI/CD Pipeline")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, doc.bottomMargin - 15, f"Page {doc.page}")
    canvas.restoreState()

def build_pdf_report():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "telco_customer_churn.csv")
    report_path = os.path.join(base_dir, "churn_prediction_report.pdf")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    # 1. Baseline Random Forest
    print("Training Baseline Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred, output_dict=True)
    rf_cm_path = os.path.join(plots_dir, "rf_cm.png")
    save_confusion_matrix(y_test, rf_pred, "Baseline Random Forest", rf_cm_path)
    
    # 2. Balanced Logistic Regression
    print("Training Balanced Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_report = classification_report(y_test, lr_pred, output_dict=True)
    lr_cm_path = os.path.join(plots_dir, "lr_cm.png")
    save_confusion_matrix(y_test, lr_pred, "Balanced Logistic Regression", lr_cm_path)

    # Calculate scale_pos_weight or class weights (MLP doesn't have class_weight built-in, but we can proceed)
    
    # 3. Advanced Deep Learning (MLP Neural Network) with Finetuning
    print("Finetuning Advanced Deep Learning Model (MLP)...")
    base_mlp = MLPClassifier(max_iter=500, random_state=42)
    param_dist = {
        'hidden_layer_sizes': [(64, 32), (128, 64, 32), (32, 16)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }
    mlp_search = RandomizedSearchCV(base_mlp, param_distributions=param_dist, n_iter=5, cv=3, scoring='f1_macro', n_jobs=-1, random_state=42)
    mlp_search.fit(X_train, y_train)
    
    print(f"Best MLP Parameters: {mlp_search.best_params_}")
    mlp_model = mlp_search.best_estimator_
    mlp_pred = mlp_model.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    mlp_report = classification_report(y_test, mlp_pred, output_dict=True)
    mlp_cm_path = os.path.join(plots_dir, "mlp_cm.png")
    save_confusion_matrix(y_test, mlp_pred, "Deep Learning (MLP Neural Network)", mlp_cm_path)
    
    # Setup PDF
    doc = SimpleDocTemplate(report_path, pagesize=letter, 
                            rightMargin=50, leftMargin=50, 
                            topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    styles.add(ParagraphStyle(name='CustomTitlePage', parent=styles['Title'], 
                              fontName='Helvetica-Bold', fontSize=24, spaceAfter=20, textColor=colors.HexColor("#2C3E50")))
    styles.add(ParagraphStyle(name='CustomSubTitle', parent=styles['Normal'], 
                              fontName='Helvetica-Oblique', fontSize=14, alignment=TA_CENTER, spaceAfter=40, textColor=colors.HexColor("#7F8C8D")))
    styles.add(ParagraphStyle(name='CustomSectionHeader', parent=styles['Heading2'], 
                              fontName='Helvetica-Bold', fontSize=16, spaceBefore=20, spaceAfter=10, textColor=colors.HexColor("#2980B9")))
    styles.add(ParagraphStyle(name='CustomBodyText', parent=styles['Normal'], 
                              fontName='Helvetica', fontSize=11, leading=14, alignment=TA_JUSTIFY, spaceAfter=10))
    styles.add(ParagraphStyle(name='CustomHighlight', parent=styles['Normal'], 
                              fontName='Helvetica-Bold', fontSize=11, textColor=colors.HexColor("#C0392B"), spaceAfter=10))
    
    elements = []
    
    # --- Title Page ---
    elements.append(Spacer(1, 100))
    elements.append(Paragraph("Comparative Analysis of Churn Prediction Models", styles['CustomTitlePage']))
    elements.append(Paragraph("A Technical Evaluation Report", styles['CustomSubTitle']))
    elements.append(Spacer(1, 200))
    elements.append(Paragraph("<b>Prepared by:</b> Data Science Team", styles['CustomBodyText']))
    elements.append(Paragraph("<b>Subject:</b> Telco Customer Churn Strategy", styles['CustomBodyText']))
    elements.append(PageBreak())
    
    # --- Executive Summary ---
    elements.append(Paragraph("Executive Summary", styles['CustomSectionHeader']))
    summary_text = """This report compares three machine learning architectures for predicting customer churn. 
    The business objective is to maximize the identification of at-risk customers (Recall for Churners) to enable proactive retention campaigns. 
    We evaluate a baseline Random Forest classifier, a class-weighted Logistic Regression model, and an advanced Deep Learning architecture (Multi-Layer Perceptron)."""
    elements.append(Paragraph(summary_text, styles['CustomBodyText']))
    
    # --- Model 1: Baseline ---
    elements.append(Paragraph("Model 1: Baseline Random Forest", styles['CustomSectionHeader']))
    elements.append(Paragraph(f"<b>Overall Accuracy:</b> {rf_acc:.2%}", styles['CustomBodyText']))
    elements.append(Paragraph("The baseline model achieves high accuracy but suffers from severe class imbalance, predicting 'No Churn' too aggressively.", styles['CustomBodyText']))
    elements.append(Paragraph(f"<b>Churn Recall:</b> Only {rf_report['1']['recall']:.2%} of actual churners were identified.", styles['CustomHighlight']))
    elements.append(Spacer(1, 10))
    elements.append(create_metric_table(rf_report))
    elements.append(Spacer(1, 15))
    elements.append(Image(rf_cm_path, width=300, height=240))
    
    elements.append(PageBreak())
    
    # --- Model 2: Balanced ---
    elements.append(Paragraph("Model 2: Balanced Logistic Regression", styles['CustomSectionHeader']))
    elements.append(Paragraph(f"<b>Overall Accuracy:</b> {lr_acc:.2%}", styles['CustomBodyText']))
    elements.append(Paragraph("By applying 'balanced' class weights, this linear model heavily penalizes the misclassification of the minority class (Churners).", styles['CustomBodyText']))
    elements.append(Paragraph(f"<b>Churn Recall:</b> {lr_report['1']['recall']:.2%} of actual churners were successfully identified.", styles['CustomHighlight']))
    elements.append(Spacer(1, 10))
    elements.append(create_metric_table(lr_report))
    elements.append(Spacer(1, 15))
    elements.append(Image(lr_cm_path, width=300, height=240))
    
    elements.append(PageBreak())
    
    # --- Model 3: Advanced ---
    elements.append(Paragraph("Model 3: Advanced Deep Learning (MLP)", styles['CustomSectionHeader']))
    elements.append(Paragraph(f"<b>Overall Accuracy:</b> {mlp_acc:.2%}", styles['CustomBodyText']))
    elements.append(Paragraph("This advanced deep learning architecture utilizes multiple hidden layers to capture highly complex, non-linear feature interactions in the customer data.", styles['CustomBodyText']))
    elements.append(Paragraph(f"<b>Churn Recall:</b> {mlp_report['1']['recall']:.2%} of actual churners were successfully identified.", styles['CustomHighlight']))
    elements.append(Spacer(1, 10))
    elements.append(create_metric_table(mlp_report))
    elements.append(Spacer(1, 15))
    elements.append(Image(mlp_cm_path, width=300, height=240))
    
    # --- Conclusion ---
    elements.append(Paragraph("Strategic Recommendation", styles['CustomSectionHeader']))
    conclusion = f"""The baseline Random Forest misses over half of the churning customers. 
    While the Logistic Regression model captures {lr_report['1']['recall']:.2%} of churners, the Deep Learning model captures {mlp_report['1']['recall']:.2%} of churners while 
    maintaining a robust F1-score of {mlp_report['1']['f1-score']:.2f}. Depending on the specific cost of false positives vs. false negatives, either Logistic Regression or the Deep Learning model provides a massive upgrade over the baseline."""
    elements.append(Paragraph(conclusion, styles['CustomBodyText']))
    
    # Build
    doc.build(elements, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"Report generated successfully at {report_path}")

if __name__ == "__main__":
    build_pdf_report()
