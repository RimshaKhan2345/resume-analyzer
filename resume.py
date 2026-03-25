import streamlit as st
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import os
import json

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)

stop_words = set(stopwords.words('english'))

# ══════════════════════════════════════════════════════════
#  CUSTOM CSS  — Dark professional theme
# ══════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --bg:      #0b0f1a;
        --surface: #111827;
        --surface2:#1a2236;
        --border:  #1f2d45;
        --accent:  #6366f1;
        --accent2: #06b6d4;
        --green:   #10b981;
        --red:     #ef4444;
        --yellow:  #f59e0b;
        --text:    #e2e8f0;
        --muted:   #1a2236;
        --soft:    #94b3c9;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .main .block-container {
        padding: 0 2rem 4rem 2rem !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }

    /* Hero */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border-bottom: 1px solid var(--border);
        padding: 48px 40px 36px;
        margin: 0 -2rem 2.5rem -2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.4);
        color: #a5b4fc;
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        letter-spacing: 2px;
        padding: 5px 14px;
        border-radius: 4px;
        margin-bottom: 16px;
        text-transform: uppercase;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 12px;
        background: linear-gradient(135deg, #ffffff 40%, #a5b4fc 70%, #67e8f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-size: 1rem;
        color: var(--soft);
        max-width: 540px;
        line-height: 1.7;
    }
    .hero-stats {
        display: flex;
        gap: 24px;
        margin-top: 24px;
        flex-wrap: wrap;
    }
    .hero-stat {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        color: var(--soft);
    }
    .hero-dot { width:8px;height:8px;border-radius:50%; }

    /* Section labels */
    .sec-label {
        font-family: 'DM Mono', monospace;
        font-size: 20px;
        letter-spacing: 3px;
        color: rgb(8 8 8);
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .sec-heading {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 6px;
    }
    .sec-sub {
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 20px;
    }

    /* Cards */
    .card-title {
        font-size: 15px;
        font-weight: 600;
        color: var(--muted);
        margin-bottom: 4px;
    }
    .card-sub {
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 16px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--surface2) !important;
        border: 2px dashed var(--border) !important;
        border-radius: 12px !important;
        transition: border-color 0.2s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
    }

    /* Text area */
    textarea {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        transition: border-color 0.2s !important;
    }
    textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
        outline: none !important;
    }

    /* Button */
    [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px 36px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
        width: 100% !important;
    }
    [data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(99,102,241,0.5) !important;
    }

    /* Metric */
    [data-testid="stMetric"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 18px 20px !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 11px !important;
        color: var(--muted) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--text) !important;
    }

    /* Progress */
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #6366f1, #06b6d4) !important;
        border-radius: 50px !important;
        height: 8px !important;
    }
    [data-testid="stProgress"] > div {
        background: var(--border) !important;
        border-radius: 50px !important;
        height: 8px !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        margin-bottom: 8px !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: var(--text) !important;
        padding: 14px 18px !important;
    }

    /* Alerts */
    [data-testid="stSuccess"] {
        background: rgba(16,185,129,0.1) !important;
        border: 1px solid rgba(16,185,129,0.3) !important;
        border-radius: 10px !important;
        color: #6ee7b7 !important;
    }
    [data-testid="stWarning"] {
        background: rgba(245,158,11,0.1) !important;
        border: 1px solid rgba(245,158,11,0.3) !important;
        border-radius: 10px !important;
        color: #fcd34d !important;
    }
    [data-testid="stError"] {
        background: rgba(239,68,68,0.1) !important;
        border: 1px solid rgba(239,68,68,0.3) !important;
        border-radius: 10px !important;
        color: #fca5a5 !important;
    }
    [data-testid="stInfo"] {
        background: rgba(99,102,241,0.1) !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
        border-radius: 10px !important;
        color: #a5b4fc !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding: 2rem 1.2rem !important;
    }

    hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

    /* Custom components */
    .score-ring-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 16px 0;
    }
    .score-ring { position: relative; width: 150px; height: 150px; margin-bottom: 12px; }
    .score-ring svg { transform: rotate(-90deg); }
    .score-ring-num {
        position: absolute; inset: 0;
        display: flex; align-items: center;
        justify-content: center; flex-direction: column;
    }
    .snum  { font-size: 2rem; font-weight: 700; color: #fff; line-height: 1; }
    .slabel { font-size: 10px; color: var(--muted); letter-spacing: 1px;
               text-transform: uppercase; margin-top: 4px;
               font-family: 'DM Mono', monospace; }
    .score-verdict {
        font-size: 13px; font-weight: 600; text-align: center;
        padding: 6px 18px; border-radius: 50px;
    }

    .skill-tag {
        display: inline-block;
        padding: 5px 14px; border-radius: 50px;
        font-size: 12px; font-weight: 600; margin: 3px;
        font-family: 'DM Mono', monospace;
    }
    .skill-match {
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.35);
        color: #6ee7b7;
    }
    .skill-miss {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.3);
        color: #fca5a5;
    }

    .rec-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 10px;
        border-left: 4px solid;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .rec-card:hover { transform: translateX(4px); box-shadow: 0 4px 20px rgba(0,0,0,0.2); }
    .rec-card.high   { border-left-color: #ef4444; }
    .rec-card.medium { border-left-color: #f59e0b; }
    .rec-card.low    { border-left-color: #10b981; }
    .rec-priority {
        font-family: 'DM Mono', monospace;
        font-size: 10px; letter-spacing: 2px;
        text-transform: uppercase; margin-bottom: 6px;
    }
    .rec-card.high   .rec-priority { color: #f87171; }
    .rec-card.medium .rec-priority { color: #fbbf24; }
    .rec-card.low    .rec-priority { color: #34d399; }
    .rec-title  { font-size: 14px; font-weight: 600; color: #e2e8f0; margin-bottom: 8px; }
    .rec-detail { font-size: 13px; color: var(--soft); line-height: 1.6; }

    .stat-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
    .stat-pill {
        background: var(--surface2); border: 1px solid var(--border);
        border-radius: 10px; padding: 14px 18px;
        text-align: center; flex: 1; min-width: 90px;
    }
    .sp-val   { font-size: 1.5rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }
    .sp-label { font-size: 10px; color: var(--muted); text-transform: uppercase;
                letter-spacing: 1px; font-family: 'DM Mono', monospace; }

    .checklist-item {
        display: flex; align-items: flex-start; gap: 12px;
        padding: 12px 16px;
        background: var(--surface2); border: 1px solid var(--border);
        border-radius: 10px; margin-bottom: 8px;
        font-size: 13px; color: var(--soft); line-height: 1.5;
    }
    .ci-box {
        width: 18px; height: 18px;
        border: 2px solid var(--border); border-radius: 5px;
        flex-shrink: 0; margin-top: 1px;
    }

    label, [data-testid="stWidgetLabel"] p {
        font-size: 13px !important; font-weight: 500 !important;
        color: var(--soft) !important; margin-bottom: 6px !important;
    }
    h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: var(--text) !important; }
    [data-testid="stCaptionContainer"] p { color: var(--muted) !important; font-size: 12px !important; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  HTML COMPONENTS
# ══════════════════════════════════════════════════════════
def hero():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">◈ AI-Powered Career Tool</div>
        <div class="hero-title">AI Resume Analyzer</div>
        <div class="hero-sub">
            Upload your resume and paste a job description — get an instant match score,
            skill gap analysis, and personalized recommendations to land the job.
        </div>
        <div class="hero-stats">
            <div class="hero-stat">
                <div class="hero-dot" style="background:#6366f1"></div>
                TF-IDF + Cosine Similarity
            </div>
            <div class="hero-stat">
                <div class="hero-dot" style="background:#06b6d4"></div>
                NLP Text Processing
            </div>
            <div class="hero-stat">
                <div class="hero-dot" style="background:#10b981"></div>
                Real-time Skill Analysis
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def score_ring(score):
    if score >= 70:
        color   = "#10b981"
        verdict = "🟢 Excellent Match"
        vstyle  = "background:rgba(16,185,129,0.12);color:#6ee7b7;border:1px solid rgba(16,185,129,0.3)"
    elif score >= 50:
        color   = "#f59e0b"
        verdict = "🟡 Moderate Match"
        vstyle  = "background:rgba(245,158,11,0.1);color:#fcd34d;border:1px solid rgba(245,158,11,0.3)"
    else:
        color   = "#ef4444"
        verdict = "🔴 Low Match"
        vstyle  = "background:rgba(239,68,68,0.1);color:#fca5a5;border:1px solid rgba(239,68,68,0.3)"

    r     = 60
    circ  = 2 * 3.14159 * r
    dash  = (score / 100) * circ
    gap   = circ - dash

    st.markdown(f"""
    <div class="score-ring-wrap">
        <div class="score-ring">
            <svg width="150" height="150" viewBox="0 0 150 150">
                <circle cx="75" cy="75" r="{r}" fill="none"
                    stroke="#1f2d45" stroke-width="10"/>
                <circle cx="75" cy="75" r="{r}" fill="none"
                    stroke="{color}" stroke-width="10"
                    stroke-dasharray="{dash:.1f} {gap:.1f}"
                    stroke-linecap="round"/>
            </svg>
            <div class="score-ring-num">
                <span class="snum" style="color:{color}">{score}%</span>
                <span class="slabel">Match</span>
            </div>
        </div>
        <div class="score-verdict" style="{vstyle}">{verdict}</div>
    </div>
    """, unsafe_allow_html=True)


def skill_chips(skills, kind="match"):
    cls  = "skill-match" if kind == "match" else "skill-miss"
    icon = "✓" if kind == "match" else "+"
    chips = "".join(f'<span class="skill-tag {cls}">{icon} {s}</span>' for s in skills)
    st.markdown(f'<div style="line-height:2.2">{chips}</div>', unsafe_allow_html=True)


def rec_card(level, title, detail):
    m  = {"HIGH": ("high","● HIGH PRIORITY"), "MEDIUM": ("medium","● MEDIUM PRIORITY"), "LOW": ("low","● LOW PRIORITY")}
    cls, label = m.get(level, ("medium","● PRIORITY"))
    st.markdown(f"""
    <div class="rec-card {cls}">
        <div class="rec-priority">{label}</div>
        <div class="rec-title">{title}</div>
        <div class="rec-detail">{detail}</div>
    </div>
    """, unsafe_allow_html=True)


def stat_pills(matched_n, missing_n, words, score):
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill">
            <div class="sp-val" style="color:#6366f1">{score}%</div>
            <div class="sp-label">Match Score</div>
        </div>
        <div class="stat-pill">
            <div class="sp-val" style="color:#10b981">{matched_n}</div>
            <div class="sp-label">Skills Matched</div>
        </div>
        <div class="stat-pill">
            <div class="sp-val" style="color:#ef4444">{missing_n}</div>
            <div class="sp-label">Skills Missing</div>
        </div>
        <div class="stat-pill">
            <div class="sp-val" style="color:#94a3b8">{words}</div>
            <div class="sp-label">Resume Words</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def checklist_item(text):
    st.markdown(f"""
    <div class="checklist-item">
        <div class="ci-box"></div>
        <span>{text}</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ══════════════════════════════════════════════════════════

# ----- Free AI Analysis (No API Required) -----
def analyze_with_ai(resume_text, job_description):
    """
    Free AI-powered resume analysis using local NLP.
    Returns dict with fields: name, skills, experience_years, match_score, summary, recommendation
    No API key required - uses rule-based extraction.
    """
    try:
        # Extract candidate name (look for common name patterns)
        name = extract_name(resume_text)
        
        # Extract skills using our existing function
        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)
        skills_list = list(resume_skills & job_skills)
        
        # Extract experience years
        exp_years = extract_experience_years(resume_text)
        
        # Calculate match score using our weighted system
        match_score = get_match_score(resume_text, job_description)
        
        # Generate summary
        summary = generate_profile_summary(resume_text, skills_list, exp_years)
        
        # Generate recommendation
        recommendation = generate_recommendation_decision(match_score, skills_list, job_skills)
        
        return {
            "name": name,
            "skills": skills_list[:10],  # Top 10 skills
            "experience_years": exp_years,
            "match_score": match_score,
            "summary": summary,
            "recommendation": recommendation
        }
    except Exception as e:
        return None


def extract_name(text):
    """Extract candidate name from resume using heuristics."""
    lines = text.split('\n')
    # Look for name at the top (usually first non-empty line)
    for line in lines[:5]:
        line = line.strip()
        if line and len(line) < 50 and not any(x in line.lower() for x in ['@', 'http', 'www', 'phone', 'cell', 'mobile']):
            # Check if it looks like a name (capitalized words)
            words = line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() if w else False for w in words):
                return line
    return "Not found"


def extract_experience_years(text):
    """Extract total years of experience from resume."""
    text_lower = text.lower()
    
    # Look for patterns like "5+ years", "3 years", "5 yrs"
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp\.?)',
        r'(?:experience|exp\.?)\s*(?:of\s*)?(\d+)\+?\s*(?:years?|yrs?)',
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:in\s*)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            return int(matches[0])
    
    # Look for date ranges in work experience
    year_pattern = r'(20\d{2}|19\d{2})\s*[-–to]+\s*(20\d{2}|19\d{2}|present|now)'
    date_ranges = re.findall(year_pattern, text_lower)
    
    if date_ranges:
        total_years = 0
        for start, end in date_ranges:
            start_year = int(start)
            if end.lower() in ['present', 'now']:
                end_year = 2025
            else:
                end_year = int(end)
            total_years += end_year - start_year
        return min(total_years, 30)  # Cap at 30 years
    
    return "Not found"


def generate_profile_summary(resume_text, skills, exp_years):
    """Generate a brief profile summary from resume."""
    text_lower = resume_text.lower()
    
    # Determine seniority
    if isinstance(exp_years, int):
        if exp_years >= 8:
            seniority = "Senior"
        elif exp_years >= 4:
            seniority = "Mid-level"
        elif exp_years >= 2:
            seniority = "Junior"
        else:
            seniority = "Entry-level"
    else:
        seniority = "Motivated"
    
    # Get top skills
    top_skills = skills[:3] if skills else ["professional"]
    skills_str = ", ".join(top_skills[:-1]) + " and " + top_skills[-1] if len(top_skills) > 1 else top_skills[0]
    
    # Detect domain focus
    domain = ""
    if any(s in text_lower for s in ["machine learning", "ai", "deep learning", "nlp"]):
        domain = "AI/ML"
    elif any(s in text_lower for s in ["web", "react", "django", "flask"]):
        domain = "Web Development"
    elif any(s in text_lower for s in ["data", "analytics", "sql", "database"]):
        domain = "Data Engineering"
    
    summary = f"{seniority} professional"
    if domain:
        summary += f" specializing in {domain}"
    if skills:
        summary += f" with expertise in {skills_str}"
    summary += "."
    
    return summary


def generate_recommendation_decision(score, matched_skills, job_skills):
    """Generate hiring recommendation based on score and skills."""
    if not job_skills:
        return "Review Manually"
    
    skills_match_ratio = len(matched_skills) / len(job_skills) if job_skills else 0
    
    if score >= 70 and skills_match_ratio >= 0.6:
        return "Shortlist"
    elif score >= 50 and skills_match_ratio >= 0.4:
        return "Maybe"
    else:
        return "Reject"
    
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text   = ""
    for page in reader.pages:
        ex = page.extract_text()
        if ex:
            text += ex
    return text


def clean_text(text):
    text  = text.lower()
    text  = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    text  = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)


def get_match_score(resume_text, job_text):
    """
    Calculate weighted match score using multiple factors:
    - Skills match: 40%
    - Experience keywords: 30%
    - TF-IDF similarity: 20%
    - Job-specific terms: 10%
    """
    # 1. Skills Score (40%)
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)
    
    if job_skills:
        skills_score = len(resume_skills & job_skills) / len(job_skills) * 100
    else:
        skills_score = 0
    
    # 2. Experience Level Score (30%)
    exp_patterns = {
        'senior': ['senior', 'sr ', 'lead', 'principal', 'staff'],
        'mid': ['mid', 'middle', 'ii', '2+', '3+', '4+', '5+'],
        'junior': ['junior', 'jr ', 'entry', 'intern', 'graduate'],
    }
    job_lower = job_text.lower()
    resume_lower = resume_text.lower()
    
    exp_matches = 0
    exp_total = 0
    for level, patterns in exp_patterns.items():
        job_has_level = any(p in job_lower for p in patterns)
        resume_has_level = any(p in resume_lower for p in patterns)
        if job_has_level:
            exp_total += 1
            if resume_has_level:
                exp_matches += 1
    
    # Also check for years of experience
    years_job = re.findall(r'(\d+)\+?\s*(?:years?|yrs?|years?)', job_lower)
    years_resume = re.findall(r'(\d+)\+?\s*(?:years?|yrs?|years?)', resume_lower)
    if years_job and years_resume:
        exp_total += 1
        if int(years_resume[0]) >= int(years_job[0]) * 0.8:  # 80% of required years
            exp_matches += 1
    
    exp_score = (exp_matches / exp_total * 100) if exp_total > 0 else 50
    
    # 3. TF-IDF Similarity (20%)
    r_clean = clean_text(resume_text)
    j_clean = clean_text(job_text)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=500)
    mat = vec.fit_transform([r_clean, j_clean])
    tfidf_sim = cosine_similarity(mat[0:1], mat[1:2])[0][0]
    tfidf_score = tfidf_sim * 100
    
    # 4. Job-Specific Keywords Score (10%)
    job_words = [w for w in j_clean.split() if len(w) > 4]
    resume_words = set(r_clean.split())
    keyword_matches = sum(1 for w in job_words[:50] if w in resume_words)
    keyword_score = (keyword_matches / min(50, len(job_words))) * 100 if job_words else 0
    
    # Weighted final score
    final_score = (
        skills_score * 0.40 +
        exp_score * 0.30 +
        tfidf_score * 0.20 +
        keyword_score * 0.10
    )
    
    # Boost score if strong skills match
    if skills_score >= 60:
        final_score = min(100, final_score * 1.1)
    
    return round(final_score, 2)


TECH_SKILLS = [
    # Programming Languages
    "python", "java", "c++", "c", "javascript", "typescript", "go", "rust", "scala", "kotlin",
    "swift", "ruby", "php", "r", "matlab", "shell", "bash", "powershell",
    # Data Science & ML
    "machine learning", "data science", "deep learning", "artificial intelligence", "ai",
    "nlp", "natural language processing", "computer vision", "data analysis", "statistics",
    "tensorflow", "keras", "pytorch", "sklearn", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "plotly", "jupyter", "spark", "hadoop",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle", "nosql",
    "database", "data warehouse", "etl",
    # Web Development
    "html", "css", "react", "angular", "vue", "node.js", "nodejs", "express",
    "flask", "django", "fastapi", "rest api", "graphql", "websocket",
    # Cloud & DevOps
    "aws", "azure", "gcp", "cloud", "docker", "kubernetes", "k8s", "ci/cd",
    "jenkins", "gitlab", "github actions", "terraform", "ansible", "linux",
    # Tools & Practices
    "git", "github", "gitlab", "bitbucket", "agile", "scrum", "kanban",
    "tdd", "unit testing", "integration testing", "ci", "cd", "devops",
    # Soft Skills
    "communication", "leadership", "teamwork", "problem solving", "analytical thinking",
    "project management", "mentoring", "collaboration",
    # Business Tools
    "excel", "powerbi", "tableau", "looker", "jira", "confluence", "slack"
]

# Skill synonyms for semantic matching
SKILL_SYNONYMS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ds": "data science",
    "da": "data analysis",
    "k8s": "kubernetes",
    "ci/cd": "ci",
    "rest": "rest api",
    "api": "rest api",
    "cloud computing": "cloud",
    "data engineering": "etl",
    "data engineer": "etl",
    "full stack": "fullstack",
    "full-stack": "fullstack",
    "oopp": "oop",
    "microservices": "micro services",
    "serverless": "cloud",
    "lambda": "aws",
}

def extract_skills(text):
    """Extract skills from text using direct matching and synonym expansion."""
    text_lower = text.lower()
    found_skills = set()
    
    # Direct skill matching
    for skill in TECH_SKILLS:
        if skill in text_lower:
            found_skills.add(skill)
    
    # Synonym expansion - map abbreviations to full skill names
    for synonym, canonical in SKILL_SYNONYMS.items():
        if synonym in text_lower and canonical in TECH_SKILLS:
            found_skills.add(canonical)
    
    return found_skills


def analyze_skills(resume_text, job_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)
    
    matched = list(resume_skills & job_skills)
    missing = list(job_skills - resume_skills)
    
    # Sort by relevance (longer/more specific skills first)
    matched.sort(key=lambda x: (-len(x.split()), x))
    missing.sort(key=lambda x: (-len(x.split()), x))
    
    return matched, missing


def generate_recommendations(resume_text, job_text, score, missing_skills):
    recs      = []
    rl        = resume_text.lower()
    jl        = job_text.lower()
    job_words = [w for w in jl.split() if w not in stop_words and len(w) > 4]
    res_set   = set(rl.split())
    top_miss  = [w for w, _ in Counter(w for w in job_words if w not in res_set).most_common(6)]

    # Adjusted thresholds for new weighted scoring system
    if score < 50:
        recs.append(("HIGH",
            "Major keyword alignment needed",
            "Your score is below 50%. Rewrite your bullet points using the exact words from the "
            "job description. If the job says 'data analysis', use those exact words — not 'data study'."))
    elif score < 70:
        recs.append(("MEDIUM",
            "Add more job-specific keywords",
            "Pick the top 5 most repeated words from the job description and ensure each appears "
            "at least once in your resume — in your skills, summary, or experience bullets."))
    else:
        recs.append(("LOW",
            "Quantify your achievements to stand out",
            "Your score is strong! Add numbers: 'improved accuracy by 15%', 'reduced time by 30%'. "
            "Quantified results make your resume stand out significantly."))

    if missing_skills:
        s = ", ".join(missing_skills[:5])
        recs.append(("HIGH",
            f"Add missing skills: {s}",
            f"The job requires <strong>{s}</strong> but these are absent from your resume. "
            "Add them to your Skills section. If you don't have them, a short online course can help."))

    if top_miss:
        w = ", ".join(top_miss)
        recs.append(("MEDIUM",
            "Important job keywords not found in your resume",
            f"These words appear frequently in the job post but not your resume: <strong>{w}</strong>. "
            "Naturally work them into your experience descriptions or summary."))

    has_summary = any(p in rl for p in ["summary","objective","profile","about me","overview"])
    if not has_summary:
        recs.append(("MEDIUM",
            "Add a Professional Summary section",
            "A 2–3 sentence summary at the top is a prime spot for job keywords. "
            "Example: 'Data Scientist with 2 years of Python and ML experience, "
            "seeking to apply NLP skills in an AI-driven team.'"))

    if any(w in jl for w in ["machine learning","deep learning","ai","artificial intelligence"]):
        if "project" not in rl and "github" not in rl:
            recs.append(("MEDIUM",
                "Add AI/ML projects or a GitHub link",
                "This is an AI/ML role — employers expect real projects. Add a Projects section "
                "with 1–2 examples and include your GitHub link. Even a small classifier project matters."))

    if any(w in jl for w in ["sql","database","mysql","postgresql"]):
        if not any(w in rl for w in ["sql","mysql","database","query"]):
            recs.append(("HIGH",
                "SQL is required but missing from your resume",
                "SQL appears in the job requirements but not your resume. Add it immediately if "
                "you know it. If not, try SQLZoo or Mode SQL Tutorial — learnable in 1–2 weeks."))

    wc = len(resume_text.split())
    if wc < 200:
        recs.append(("HIGH",
            f"Resume is too short ({wc} words) — expand it",
            "A strong resume has 400–700 words. Expand your experience bullets: describe WHAT "
            "you did, HOW you did it, and the RESULT. Aim for 3–5 bullets per role."))
    elif wc > 1000:
        recs.append(("MEDIUM",
            f"Resume may be too long ({wc} words) — consider trimming",
            "For under 10 years experience, aim for 1 page (~500 words). "
            "Remove old jobs and cut weak bullet points."))

    return recs


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
def build_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:8px 0 20px">
            <div style="font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:4px">🎯 ResumeAI</div>
            <div style="font-size:11px;color:#64748b;font-family:'DM Mono',monospace;letter-spacing:1px">
                AI-POWERED CAREER TOOL
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:2px;
                    color:#64748b;text-transform:uppercase;margin-bottom:12px">How It Works</div>
        """, unsafe_allow_html=True)

        for num, text in [("1","Upload your PDF resume"),("2","Paste the job description"),
                          ("3","Click Analyze Resume"),("4","Review score & recommendations")]:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:14px">
                <div style="width:24px;height:24px;border-radius:6px;
                            background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);
                            display:flex;align-items:center;justify-content:center;
                            font-size:11px;color:#a5b4fc;font-weight:700;
                            flex-shrink:0;font-family:'DM Mono',monospace">{num}</div>
                <div style="font-size:13px;color:#94a3b8;padding-top:3px;line-height:1.4">{text}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:10px;letter-spacing:2px;
                    color:#64748b;text-transform:uppercase;margin-bottom:12px">Score Guide</div>
        """, unsafe_allow_html=True)

        for color, label, desc in [
            ("#10b981","70%+","Excellent match"),
            ("#f59e0b","50–70%","Moderate match"),
            ("#ef4444","Below 50%","Needs improvement"),
        ]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
                <div style="width:10px;height:10px;border-radius:50%;background:{color};flex-shrink:0"></div>
                <span style="font-size:13px;font-weight:600;color:#e2e8f0">{label}</span>
                <span style="font-size:12px;color:#64748b">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:11px;color:#374151;line-height:1.7;text-align:center">
            Built with Python · NLTK · Streamlit<br>TF-IDF + Cosine Similarity
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="Smart AI Resume Analyzer", page_icon="🎯", layout="wide")
    inject_css()
    build_sidebar()
    hero()

    # ── Inputs ─────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="sec-label">Step 01 — Your Resume</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📄 Upload Resume</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Upload your resume as a PDF file</div>', unsafe_allow_html=True)
        pdf_file    = st.file_uploader("Resume PDF", type="pdf", label_visibility="collapsed")
        resume_text = ""
        if pdf_file:
            resume_text = extract_text_from_pdf(pdf_file)
            st.success(f"✅ Resume uploaded — {len(resume_text.split())} words extracted")
            with st.expander("👁 Preview extracted text"):
                st.markdown(f"""
                <div style="font-size:12px;color:#64748b;line-height:1.8;
                            font-family:'DM Mono',monospace;
                            background:#0b0f1a;padding:14px;border-radius:8px;
                            white-space:pre-wrap">{resume_text[:800]}...</div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="sec-label">Step 02 — Target Job</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💼 Job Description</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Paste the full job posting here</div>', unsafe_allow_html=True)
        job_desc = st.text_area(
            "Job Description",
            height=240,
            placeholder="Paste the job description here...\n\nE.g: We are looking for a Python Developer with experience in Machine Learning...",
            label_visibility="collapsed"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Analyze Button ─────────────────────────────────
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        analyze = st.button("🔍 Analyze My Resume", use_container_width=True)

    if analyze:
        if not resume_text or not job_desc.strip():
            st.warning("⚠️ Please upload a resume AND paste a job description before analyzing.")
            return

        with st.spinner("Analyzing your resume..."):
            score            = get_match_score(resume_text, job_desc)
            matched, missing = analyze_skills(resume_text, job_desc)
            recs             = generate_recommendations(resume_text, job_desc, score, missing)
            word_count       = len(resume_text.split())

            #----- FREE AI ANALYSIS (No API Required) -----
            ai_result = analyze_with_ai(resume_text, job_desc)

        st.markdown("---")

        # ── Results Header ─────────────────────────────
        st.markdown("""
        <div class="sec-label">Analysis Results</div>
        <div class="sec-heading">Here's how your resume performs 📊</div>
        """, unsafe_allow_html=True)

        stat_pills(len(matched), len(missing), word_count, score)

        # ── Score Ring + Skills ─────────────────────────
        ring_col, skills_col = st.columns([1, 2], gap="large")

        with ring_col:
            st.markdown('<div class="sec-label">Match Score</div>', unsafe_allow_html=True)
            score_ring(score)
            st.progress(int(min(score, 100)))

        with skills_col:
            st.markdown('<div class="sec-label">Skills Breakdown</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:13px;font-weight:600;color:#6ee7b7;margin-bottom:8px">
                ✅ Matched Skills
            </div>
            """, unsafe_allow_html=True)
            if matched:
                skill_chips(matched, "match")
            else:
                st.markdown('<div style="font-size:13px;color:#64748b;padding:8px 0">No matching skills found in job description</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            st.markdown("""
            <div style="font-size:16px;font-weight:600;color:rgb(252, 85, 75);margin-bottom:8px">
                ❌ Missing Skills
            </div>
            """, unsafe_allow_html=True)
            if missing:
                skill_chips(missing, "miss")
            else:
                st.markdown('<div style="font-size:18px;color:rgb(52, 211, 153);padding:8px 0">🎉 None — you have all the key skills!</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Recommendations ─────────────────────────────
        st.markdown("""
        <div class="sec-label">Personalized Recommendations</div>
        <div class="sec-heading">💡 What to improve</div>
        <div class="sec-sub">Based on the job description — here's exactly what to fix in your resume</div>
        """, unsafe_allow_html=True)

        for r in recs:
            level = "HIGH" if "HIGH" in r[0] else ("LOW" if "LOW" in r[0] else "MEDIUM")
            rec_card(level, r[1], r[2])

        st.markdown("---")

        # ── Checklist ───────────────────────────────────
        st.markdown("""
        <div class="sec-label">Before You Apply</div>
        <div class="sec-heading">📋 Quick Action Checklist</div>
        <div class="sec-sub">Complete these steps to maximize your chances</div>
        """, unsafe_allow_html=True)

        items = []
        if score < 50:
            items.append("Rewrite 3+ experience bullets using keywords directly from the job description")
        if missing:
            items.append(f"Add to your Skills section: {', '.join(missing[:4])}")
        if "github" not in resume_text.lower():
            items.append("Add your GitHub profile link — critical for tech roles")
        if "linkedin" not in resume_text.lower():
            items.append("Add your LinkedIn profile URL to the top of your resume")
        if word_count < 300:
            items.append("Expand experience bullets — aim for 3–5 bullet points per job role")
        items.append("Run a spell check on your entire resume before submitting")
        items.append("Save and submit as PDF format (not .docx) for clean formatting")

        for item in items:
            checklist_item(item)

        # ── Footer ──────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;padding:20px 24px;
                    background:#111827;border:1px solid #1f2d45;
                    border-radius:12px;">
            <div style="font-size:13px;color:#64748b;line-height:1.7">
                💬 <strong style="color:#94a3b8">Pro tip:</strong>
                Update your resume based on the recommendations above,
                then re-upload it here to see your improved score!
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()