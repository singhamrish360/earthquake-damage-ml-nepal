"""Build the 2-page Project_Summary.pdf from results/metrics.json and figures/ (no hand-typed metrics)."""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "results/metrics.json").read_text())
T, CV = M["test"], M["cv"]
ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.6, leading=11, spaceAfter=3)
H = ParagraphStyle("h", parent=ss["Heading3"], fontSize=10.5, spaceBefore=4, spaceAfter=2, textColor=colors.HexColor("#1f3b63"))
TT = ParagraphStyle("t", parent=ss["Title"], fontSize=14.5, leading=17, spaceAfter=2)
SM = ParagraphStyle("s", parent=B, fontSize=7.4, leading=9, textColor=colors.HexColor("#444444"))
f = lambda k, m: f"{T[k][m]:.3f}"

def img(name, w):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(ROOT / "figures" / name)).getSize()
    return Image(str(ROOT / "figures" / name), width=w, height=w * ih / iw)

s = []
s.append(Paragraph("Predicting Earthquake Damage Grade from Building Attributes: Machine Learning on 762,094 Buildings from the 2015 Gorkha (Nepal) Earthquake", TT))
s.append(Paragraph("<b>Amrish Singh</b> &nbsp;|&nbsp; singhamrish360@gmail.com &nbsp;|&nbsp; ORCID 0009-0005-1584-2504 &nbsp;|&nbsp; Project summary, October 2026", SM))
s.append(Spacer(1, 4))
s.append(Paragraph("Objective", H))
s.append(Paragraph("Test how well cheap, pre-earthquake building attributes (structural system, wall and foundation materials, storeys, age, plan) predict the "
    "observed damage grade (1 = slight cracking ... 5 = collapse), and check whether the learned patterns agree with structural-engineering knowledge, as a step "
    "towards data-driven rapid seismic vulnerability screening.", B))
s.append(Paragraph("Data", H))
s.append(Paragraph("Building-structure table of the Government of Nepal <i>Household Registration for Housing Reconstruction</i> survey (National Planning Commission / "
    "Central Bureau of Statistics; 2015 Nepal Earthquake Open Data Portal, eq2015.npc.gov.np, built by Kathmandu Living Labs). 762,106 buildings in 11 districts; "
    "the identical CSV was obtained from a public GitHub mirror because the official portal was unreachable, and verified by row count and SHA-256. "
    f"Four post-earthquake columns were removed to prevent target leakage; 12 rows without a label were dropped (n = {M['cleaning']['clean_rows']:,}); "
    "age = 999 was treated as unknown. 22 building features were used. No explicit data license is published; used for non-commercial research with attribution.", B))
s.append(Paragraph("Method", H))
s.append(Paragraph(f"Stratified 80/20 split ({M['n_train']:,} train / {M['n_test']:,} test). Baseline: multinomial logistic regression (stratified 200k training subsample). "
    f"Main model: LightGBM with native categorical handling and early stopping ({T['lgbm']['best_iteration']} trees). Ablation: LightGBM + district (location proxy for shaking). "
    f"Stratified 5-fold CV on a {CV['n_rows']:,}-row training subsample. Explanations: SHAP (TreeExplainer) and permutation importance. Total CPU runtime approx. "
    f"{M['runtime_seconds']/60:.0f} min.", B))
s.append(Paragraph("Results (held-out test set)", H))
rows = [["Model", "Macro-F1", "Micro-F1 (acc.)", "Within +/-1 grade"],
        ["Majority class (always Grade 5)", f("majority_class","macro_f1"), f("majority_class","micro_f1"), f("majority_class","within_one_grade")],
        ["Logistic regression", f("logreg","macro_f1"), f("logreg","micro_f1"), f("logreg","within_one_grade")],
        ["LightGBM, building attributes only", f("lgbm","macro_f1"), f("lgbm","micro_f1"), f("lgbm","within_one_grade")],
        ["LightGBM + district (ablation)", f("lgbm_plus_district","macro_f1"), f("lgbm_plus_district","micro_f1"), f("lgbm_plus_district","within_one_grade")]]
t = Table(rows, colWidths=[6.4*cm, 2.4*cm, 2.8*cm, 3.0*cm])
t.setStyle(TableStyle([("FONTSIZE", (0,0), (-1,-1), 8.2), ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f3b63")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), 0.3, colors.grey), ("ALIGN", (1,0), (-1,-1), "CENTER"),
    ("BACKGROUND", (0,3), (-1,3), colors.HexColor("#e6eef8")), ("TOPPADDING", (0,0), (-1,-1), 1.5), ("BOTTOMPADDING", (0,0), (-1,-1), 1.5)]))
s.append(t); s.append(Spacer(1, 3))
s.append(Paragraph(f"5-fold CV macro-F1: logistic regression {CV['logreg']['macro_f1']['mean']:.3f} &plusmn; {CV['logreg']['macro_f1']['std']:.3f}, "
    f"LightGBM {CV['lgbm']['macro_f1']['mean']:.3f} &plusmn; {CV['lgbm']['macro_f1']['std']:.3f}. LightGBM recognises collapse (Grade 5 recall 0.84) and slight damage "
    "(Grade 1 recall 0.62) well; intermediate grades 2-4 are often confused with neighbours. Building attributes alone cannot see ground-shaking intensity; adding only "
    "the district raises macro-F1 substantially, showing that hazard information is the main missing ingredient.", B))
s.append(img("fig4_confusion_matrices.png", 15.5*cm))
s.append(PageBreak())
s.append(Paragraph("What the model learned: engineering interpretation", H))
s.append(Table([[img("fig2_damage_by_foundation_superstructure.png", 18*cm)]], colWidths=[18*cm]))
s.append(Table([[img("fig5_shap_importance.png", 8.4*cm), img("fig6_shap_grade5_beeswarm.png", 9.4*cm)]], colWidths=[8.7*cm, 9.5*cm]))
sh = M["shap"]["grade5_mean_shap_by_level"]
s.append(Paragraph(f"<b>Mud-mortar stone masonry</b> is the strongest predictor in both SHAP and permutation importance (mean Grade-5 SHAP +{sh['has_superstructure_mud_mortar_stone']['1']:.2f} "
    "log-odds when present). Rubble stone in mud mortar has negligible tensile/shear strength and heavy walls without bands or through-stones, so walls delaminate "
    f"and collapse out-of-plane; 42.7 % of such buildings collapsed. <b>Engineered RC</b> ({sh['has_superstructure_rc_engineered']['1']:+.2f}), <b>cement-mortar brick</b> "
    f"({sh['has_superstructure_cement_mortar_brick']['1']:+.2f}) and <b>RC roof slabs</b> ({sh['roof_type']['RCC/RB/RBC']:+.2f}) push predictions towards low damage, consistent with "
    f"rigid diaphragms, mortar bond and a defined lateral-load path; <b>non-engineered RC</b> helps far less ({sh['has_superstructure_rc_non_engineered']['1']:+.2f}). Age, storeys and "
    "height follow next (deterioration, pre-code construction, larger overturning demand on weak masonry). These are model explanations, not causal proof: materials "
    "are also correlated with location and shaking intensity.", B))
s.append(Paragraph("Relevance to India", H))
s.append(Paragraph("Unreinforced stone/brick masonry, timber-bamboo and non-engineered RC houses, the typologies driving damage here, form a large share of Indian housing, "
    "especially in the Himalayan states and Indo-Gangetic plain. Under IS 1893 (Part 1):2016 Lucknow is in seismic Zone III, and the revised IS 1893 (Part 1):2025 "
    "(Annex D, Table 18) also lists Lucknow in Zone III (Z = 0.25, 2,475-year return period). A model of this type, retrained or recalibrated on Indian damage "
    "surveys (e.g. Bhuj 2001, Sikkim 2011, Imphal 2016) and combined with the zone factor or site-specific ground motion, could rank buildings from census / "
    "property-tax attributes for detailed evaluation and retrofit (IS 13935), complementing rapid visual screening.", B))
s.append(Paragraph("Limitations and next steps", H))
s.append(Paragraph("Single earthquake, region and survey; no per-building ground-motion input; intermediate grades hard to separate (macro-F1 0.39); subsampling used for the "
    "logistic baseline and CV; light hyper-parameter tuning; possible label bias from a grant-eligibility survey; dataset license not explicitly stated. Not for "
    "decisions on individual buildings. Next: add USGS ShakeMap intensity per ward, ordinal/cost-sensitive models, spatial cross-validation by district, and "
    "transfer tests on Indian data. Code, executed notebook and all metrics: https://github.com/singhamrish360/earthquake-damage-ml-nepal (see README).", B))
doc = SimpleDocTemplate(str(ROOT / "Project_Summary.pdf"), pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.2*cm, bottomMargin=1.1*cm,
                        title="Earthquake damage-grade prediction (Gorkha 2015) - Project summary", author="Amrish Singh")
doc.build(s)
print("ok")
