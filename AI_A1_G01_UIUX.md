# UI/UX Design Document — AI_A1_G01
## Musanze HarvestLink Cooperative Decision Dashboard

**Group:** AI-G01  
**Course:** SWE 3513 — Artificial Intelligence  
**Assignment:** 1  

---

## Page 1: Cover and Operational Problem

**Musanze HarvestLink Cooperative — Decision Dashboard**

**Group Code:** AI-G01  
**Team Members:** [Member 1], [Member 2], [Member 3], [Member 4], [Member 5]  
**Stakeholder:** Musanze HarvestLink Cooperative Operations Team  
**Three Decisions Supported:**
1. **Harvest Weight Estimation** — Predict expected potato yield (kg) per collection point
2. **Dispatch Attention Flagging** — Identify consignments needing priority dispatch
3. **Collection Point Grouping** — Cluster farms with similar operating profiles for resource planning

---

## Page 2: User Journey and Information Flow

**CSV Upload → Data Quality Review → Model Predictions → Human Decision**

1. **Upload:** Staff uploads daily collection CSV (schema validated automatically)
2. **Review:** Data quality report shows missing values, duplicates, statistics
3. **Predict:** Models generate yield estimates, dispatch flags, cluster assignments
4. **Decide:** Operations team reviews predictions, applies local knowledge, confirms dispatch

---

## Page 3: Annotated Wireframe — Data Quality & Predictions

**Data Quality Panel:**
- Row count, feature count, missing values table
- SHA-256 fingerprint for audit trail
- Descriptive statistics (mean, std, min, max per feature)

**Prediction Panel:**
- Table: Record ID | Predicted Yield (kg) | Confidence Interval
- Color coding: Green (high confidence), Yellow (medium), Red (low)
- Sortable by predicted yield or confidence

---

## Page 4: Annotated Wireframe — Classification & Clusters

**Dispatch Attention Panel:**
- Binary flag: 🔴 Attention Needed / 🟢 Normal
- Probability score (0-100%)
- Explanation: "High probability due to low soil pH + high distance"

**Cluster View:**
- Scatter plot (PCA) colored by cluster
- Cluster profile cards: avg plot size, avg distance, typical arrival hour
- Hover: shows member record IDs

---

## Page 5: Responsible AI States

| State | Indicator | Action |
|-------|-----------|--------|
| **Uncertainty** | Prediction interval > 20% | Flag for expert review |
| **Missing Data** | Any required field missing | Block prediction, show error |
| **Model Limitation** | Extrapolation beyond training range | Warning banner, suggest manual estimate |
| **Human Override** | Staff can edit predictions | Log override with reason |

---

## Page 6: Visual System & Rationale

**Accessibility:**
- WCAG AA contrast ratios
- Colorblind-safe palette (viridis for continuous, tab10 for categorical)
- Text labels on all color-coded elements

**Typography:** Inter font, 14px base, clear hierarchy

**Three Justified Decisions:**
1. **Tabular predictions over charts** — Staff compare exact numbers for dispatch logs
2. **Probability + binary flag** — Probability enables prioritization; flag enables quick action
3. **PCA scatter for clusters** — 2D projection balances interpretability with information retention

---

## Page 7: Implementation Notes

**Tech Stack:** React + Tailwind CSS + Chart.js  
**API:** `POST /predict` accepts JSON, returns predictions  
**Audit:** All predictions logged with timestamp, model version, input hash