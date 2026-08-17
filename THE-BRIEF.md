# The Full Capstone Brief
## Flood Risk Classifier | AI-19 | 3MTT NextGen Cohort
## Adeniji Yusuf | FE/24/5369659950 | Ogun / Grazac Technologies Limited

---

## SECTION 1: PROJECT IDENTITY

| Field | Value |
|-------|-------|
| **Fellow Name** | Adeniji Yusuf |
| **Fellow ID** | `FE/24/5369659950` |
| **Track** | AI & Machine Learning NextGen Cohort |
| **State / Training Provider** | Ogun / Grazac Technologies Limited |
| **Project Brief ID** | **AI-19** |
| **Project Title** | Flood Risk Classifier |
| **Nigerian Problem Context** | Riverine communities in Nigeria face significant vulnerabilities and urgently need localized flood-risk awareness to protect lives and property. |

---

## SECTION 2: WHAT TO BUILD (THE MVP)

Following the **MVP Philosophy** (depth over breadth — one fully functional feature beats five broken ones), build the smallest viable version that solves the core problem well in **3–4 weeks**.

### Step 1: Data Acquisition & Preprocessing
- Gather or prepare a dataset containing environmental features:
  - Average monthly rainfall
  - Elevation
  - Proximity to rivers
  - Soil type
  - Drainage infrastructure rating
- Target variable: **Flood Risk Class** (e.g., Low, Medium, High)
- Clean the data using `pandas`
- Handle missing values
- Encode categorical variables

### Step 2: Model Training
- Use `scikit-learn` to build a classification model
- Recommended algorithms: Decision Tree, Random Forest, or Logistic Regression
- Split data into training and testing sets
- Evaluate performance

### Step 3: Feature Input & Prediction Interface
- Create a simple function or interface where a user can input environmental features for a community
- Receive a predicted risk class
- Can be a Google Colab notebook form or a lightweight script

### Step 4: Model Evaluation
- Generate evaluation metrics:
  - Confusion matrix
  - Accuracy score
  - Classification report
- Demonstrate the model's reliability

---

## SECTION 3: DATA SOURCES STRATEGY

Three ways to obtain data for your project:

### 1. Key Open-Data Sources (Highly Recommended)
- **NiMet / NIHSA** (`nimet.gov.ng` / `nihsa.gov.ng`)
  - Excellent for official Nigerian rainfall, weather, and hydrological flood data
- **Humanitarian Data Exchange (HDX)** (`data.humdata.org`)
  - Search for "Nigeria flood data" or "Nigeria population and environment"
- **Kaggle** (`kaggle.com/datasets`)
  - Search for "Nigeria flood risk" or general flood prediction datasets adaptable to Nigerian context

### 2. Collect Your Own Data
- Gather local observations or historical flood records from specific communities in Ogun State
- Use short surveys or local environmental reports

### 3. Generate Documented Synthetic Data
- **Permitted** if real-world data is completely impractical to find
- **You MUST clearly document** how the synthetic data was generated
- Ensure features realistically mimic Nigerian riverine conditions

---

## SECTION 4: RULES, REGULATIONS & SUBMISSION GUIDELINES

### Individual Work
- The capstone project must be your **OWN individual solution**
- Even though other Fellows share the `AI-19` brief, duplicate or plagiarized submissions will be disqualified

### Submission Portal
- Submit final deliverables on the official **3MTT portal** for verification and scoring by the central team

### Submission Validity Checklist (Must Satisfy ALL 5)
1. **Registered Fellow:** Linked to your verified ID (`FE/24/5369659950`)
2. **Assigned Brief:** Work must directly match the `AI-19 Flood Risk Classifier` brief
3. **Working Deliverable:** The required functional files must be present
4. **Demo Video:** A **2–3 minute video** where you explain your own work and demonstrate the model. *This is a mandatory authenticity check.*
5. **Original Work:** Free from plagiarism

---

## SECTION 5: EXPECTED DELIVERABLES

| Deliverable | Description |
|-------------|-------------|
| **Code Repository** | Jupyter Notebook or GitHub repository containing your clean code |
| **Trained Model File** | The serialized model ready for inference |
| **Evaluation Results** | Metrics, confusion matrix, classification report |
| **README.md** | Clear documentation on how to run the project and detailing your data source |
| **Demo Video** | 2–3 minute video demonstration |

### Suggested Tools
- Python
- `pandas`
- `scikit-learn`
- Google Colab

---

## SECTION 6: SCORING AND QUALITY RUBRIC

Once your project passes the validity gate, it will be scored out of **100 marks** based on:

| Criteria | Weight | What Judges Look For |
|----------|--------|---------------------|
| **Adherence to Brief & Completeness** | 20% | Did you build exactly what the brief asked for? |
| **Functionality / Effectiveness** | 25% | Does the model run reliably and accurately classify risk? |
| **Technical Quality / Craft** | 15% | Clean, sensible code and good machine learning practices |
| **User Experience / Clarity** | 15% | Clear presentation of inputs, risk classes, and key factors |
| **Innovation & Nigerian-Context Fit** | 10% | How well the model addresses the specific realities of Nigerian riverine communities |
| **Documentation & Demo Video** | 15% | A clear README and an articulate video explanation |

### Certification Score Bands
| Band | Score | Meaning |
|------|-------|---------|
| **Distinction** | 80 – 100 | Exceeds expectations; portfolio-ready |
| **Pass** | 60 – 79 | Meets the standard for **3MTT Certification** |
| **Revise & Resubmit** | 40 – 59 | Requires adjustments |
| **Not Yet Met** | 0 – 39 | Does not meet standard |

---

## SECTION 7: THE HIDDEN OPPORTUNITIES

The brief asks for a "simple function or interface" and "Google Colab notebook form." But the rubric rewards:

- **User Experience (15%)** — A Google Colab form is functional but not "clear presentation." A web interface judges can click is stronger.
- **Innovation & Nigerian-Context Fit (10%)** — Using Nigerian community names, local rivers, seasonal context, and local language considerations scores here.
- **Documentation & Demo Video (15%)** — A video where you explain the *Nigerian problem* and show *your solution* in action scores higher than a screen recording of code running.

### What "Portfolio-Ready" Looks Like
- A live URL judges can test from their phone
- Visual selectors (not dropdowns) for non-technical users
- GPS auto-detection for elevation and river proximity
- Color-coded results with plain-language advisories
- A name that sounds like a real product, not a student assignment
- A demo video that tells a story, not just shows accuracy scores

---

## SECTION 8: THE COMPOSITE RISK PHILOSOPHY

The brief says "classifies a community's flood risk level based on environmental features." The default interpretation is "rainfall = flood risk." But the winning interpretation is:

> **Flood risk is composite.** Rainfall is the trigger. Elevation, river proximity, soil type, and drainage determine whether that rain becomes a disaster.

A community can survive heavy rainfall if:
- The soil is sandy (drains fast)
- The elevation is high (water flows away)
- The drainage is good (channels carry water away)
- The river is far (no backflow)

But combine **average rainfall** with:
- Clay soil (holds water)
- Low elevation (water pools)
- No drainage (water has nowhere to go)
- Close to river (backflow risk)

That is where people die. Your model must reflect this composite reality.

### Recommended Feature Set (5 Features)
| # | Feature | Why It Matters | Data Source |
|---|---------|---------------|-------------|
| 1 | Average Monthly Rainfall (mm) | The trigger | NiMet / seasonal average |
| 2 | Elevation (meters) | Low-lying areas flood first | GPS + Open Topo Data API |
| 3 | Distance to Nearest River (km) | Proximity = direct exposure | GPS + river coordinate database |
| 4 | Soil Type | Clay holds water; sand drains it | User input (visual selector) |
| 5 | Drainage Infrastructure | Channels, culverts, gutters | User input (visual selector) |

### Seasonal Context (Critical for Realism)
The same probability means different things in Dry vs. Wet season:

| Model Probability | Dry Season Label | Wet Season Label |
|-------------------|-----------------|------------------|
| 0 – 40% | Low | Low |
| 41 – 70% | Medium | Medium-High |
| 71 – 100% | High | Critical |

This seasonal weighting makes the model more actionable. Heavy rain in dry season is more dangerous than the same rain in wet season because the ground is not prepared.

---

## SECTION 9: THE USER-CENTRIC IMPERATIVE

The brief asks for a "simple function or interface." But the problem statement says communities "urgently need localized flood-risk awareness."

If your interface requires:
- A laptop
- Python knowledge
- Understanding of "probability thresholds"
- Reading a confusion matrix

Then you have built a **classroom demo**, not a solution for the actual problem.

### The Real Users
| User | Device | Literacy | Needs |
|------|--------|----------|-------|
| Mama Nkechi (farmer, 52) | Feature phone (₦5,000) | Limited formal education | "Should I move my goats tonight?" |
| Community Volunteer (28, youth corps) | Smartphone with WhatsApp | Reads English + Yoruba | "What do I tell the village head?" |
| 3MTT Judge | Laptop/phone | Technical background | "Does this actually work? Is it original?" |

### What "Simple" Actually Means
- **One-button GPS detection** — auto-fills elevation, river distance, season
- **Visual selectors** — tap what your soil looks like, tap what your drainage looks like
- **Color-coded output** — red means danger (no reading required)
- **Plain-language advisory** — "Move livestock to higher ground" not "Probability = 0.87"
- **Shareable result** — copy text for WhatsApp broadcast to the community

---

## SECTION 10: THE DEMO VIDEO FORMULA

Your 2–3 minute video is a **mandatory authenticity check**. But it is also your highest-leverage scoring opportunity.

### Structure (2 minutes 30 seconds)

**0:00 – 0:20: The Hook (The Problem)**
> "Riverine communities in Ogun State don't need a data scientist. They need to know if their livestock will survive the night. Most flood tools only look at rainfall. But I know that elevation, river proximity, and soil type matter just as much."

**0:20 – 0:50: The Product (The Solution)**
> "This is Phense. Phense is an AI-powered flood risk assessment tool for Nigerian riverine communities. A community member clicks one button — their location is detected, elevation is fetched from satellite data, and distance to the nearest river is calculated automatically. They only need to know their soil type and drainage quality — things every farmer already knows."

**0:50 – 1:20: The Demo (The Proof)**
- Show high-risk scenario: clay soil, no drainage, low elevation, wet season, close to river
- Show the result: critical risk, confidence score, primary driver, actionable advisory
- Show low-risk scenario: sandy soil, good drainage, high elevation, dry season
- Contrast: same rainfall, completely different outcome

**1:20 – 1:50: The Technical Depth**
> "Under the hood, it's a Random Forest trained on [N] records. [X]% accuracy. The top feature isn't rainfall — it's [actual top feature]. This proves the model learned the real physics of flooding, not just 'rain equals flood.'"

**1:50 – 2:10: The Innovation**
> "What makes Phense different is [your unique angle: GPS auto-detection, visual selectors, seasonal context, Nigerian river database, etc.]"

**2:10 – 2:30: The Roadmap**
> "For production, Phense connects to NiMet live data, integrates with USSD for feature phones, and outputs advisories in Yoruba. But this MVP proves the core concept: AI can make flood risk understandable and actionable for non-technical communities."

### Video Production Tips
- **Show your face** for at least 10 seconds (authenticity check)
- **Show the actual product running** — not slides, not mockups
- **Narrate live** — don't add voiceover to screen recording
- **Use a real community name** from Ogun State
- **Speak clearly** — judges may not be Nigerian
- **End with your name and Fellow ID** — "I am Adeniji Yusuf, FE/24/5369659950."

---

## SECTION 11: THE ARCHITECTURE BLUEPRINT

### Recommended Stack for "Portfolio-Ready"
```
Frontend:    HTML/CSS/JS (single page, no framework needed)
Backend:     FastAPI (lightning fast, modern, easy to deploy)
ML:          scikit-learn Random Forest
Data:        pandas for preprocessing
Deployment:  Render.com (free, Docker support, custom domain)
```

### Why Not Streamlit?
- Re-runs entire Python script on every interaction (slow)
- Looks like a prototype, not a product
- Hard to customize visual components
- Difficult to integrate GPS auto-detection cleanly
- Judges have seen 100 Streamlit apps

### Why Not Jupyter Notebook as Final Deliverable?
- The brief "suggests" Google Colab but does not mandate it
- A GitHub repo with pure Python + web UI is stronger
- Shows software engineering + ML competence
- Deployable — judges can test it live

### File Structure for Submission
```
phense/
├── data/
│   ├── nigeria_flood_data.csv
│   └── rivers.json
├── models/
│   ├── phense_model.pkl
│   ├── encoders.pkl
│   └── metrics.json
├── static/
│   └── index.html          ← The visual interface
├── train.py                ← Training pipeline
├── predict.py              ← Prediction engine
├── app.py                  ← FastAPI backend
├── Dockerfile              ← For deployment
├── requirements.txt
└── README.md               ← Documentation + data source
```

---

## SECTION 12: THE HONESTY IMPERATIVE

### Limitations to Acknowledge
1. **Synthetic Data:** If you use synthetic data, say so. Document exactly how it was generated.
2. **Accuracy Range:** 70–80% is honest and respectable. 95%+ on synthetic data is suspicious.
3. **Scope:** If your model only covers Ogun State, say so. Explain the roadmap for expansion.
4. **Season Feature:** If season shows low importance in your model, explain why (rainfall captures the effect) and how it would be stronger with real NiMet data.

### Why Honesty Scores Higher
- Judges are not fooled by inflated metrics
- Acknowledging limitations shows **maturity**
- It proves you understand your model, not just copied code
- It opens the door to discuss the roadmap

---

*Full Capstone Brief v1.0 | Extracted from official 3MTT documentation*
*Prepared for: Adeniji Yusuf | FE/24/5369659950 | AI-19 Flood Risk Classifier*
*Product Name: Phense*
