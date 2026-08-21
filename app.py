import re
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="AI ResumeRanker", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {padding: 24px; border-radius: 18px; border: 1px solid #e5e7eb; background: #f8fafc;}
.hero h1 {margin: 0; font-size: 38px;}
.small {color:#64748b;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🤖 AI ResumeRanker</h1><p class="small">Semantic resume screening and candidate ranking using Sentence Transformers.</p></div>', unsafe_allow_html=True)

SKILLS = [
"python","java","c++","javascript","typescript","sql","mysql","postgresql","mongodb",
"machine learning","deep learning","data science","natural language processing","nlp",
"computer vision","pandas","numpy","scikit-learn","tensorflow","pytorch",
"html","css","react","node.js","flask","django","streamlit",
"aws","azure","gcp","docker","kubernetes","git","github",
"excel","power bi","tableau","statistics","data visualization"
]

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(file):
    reader=PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def clean(text):
    text=text.lower()
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9+#.\s-]"," ",text)).strip()

def email(text):
    m=re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+",text)
    return m.group(0) if m else "Not found"

def phone(text):
    m=re.search(r"(?:\+91[-\s]?)?[6-9]\d{9}",text)
    return m.group(0) if m else "Not found"

def skills(text):
    t=text.lower()
    return [s for s in SKILLS if s in t]

def experience(text):
    patterns=[r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience",
              r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*years?"]
    for p in patterns:
        m=re.search(p,text.lower())
        if m:return float(m.group(1))
    return 0.0

def education(text):
    t=text.lower()
    for k,v in {"phd":"PhD","m.tech":"M.Tech","mca":"MCA","mba":"MBA",
                "b.tech":"B.Tech","btech":"B.Tech","b.e":"B.E.","bca":"BCA",
                "b.sc":"B.Sc.","bsc":"B.Sc."}.items():
        if k in t:return v
    return "Not detected"

with st.sidebar:
    st.header("⚙️ AI Settings")
    threshold=st.slider("Minimum AI match score",0,100,0)
    model_name=st.selectbox("Embedding model",["all-MiniLM-L6-v2"])
    st.caption("The model converts job descriptions and resumes into semantic embeddings.")
    st.info("Use this as decision support. Human review is required for real recruitment.")

job=st.text_area("📝 Job Description",height=220,
                 placeholder="Example: Data Scientist with Python, SQL, machine learning, pandas, NLP and 2 years experience.")

uploads=st.file_uploader("📄 Upload PDF Resumes",type=["pdf"],accept_multiple_files=True)

if st.button("🚀 Run AI Screening",type="primary",use_container_width=True):
    if not job.strip(): st.error("Enter a job description."); st.stop()
    if not uploads: st.error("Upload at least one PDF resume."); st.stop()

    with st.spinner("Loading AI model and analyzing resumes..."):
        model=load_model()
        job_embedding=model.encode(clean(job),convert_to_tensor=True)
        data=[]

        for f in uploads:
            try:
                text=extract_text(f)
                resume_skills=skills(text)
                required=[s for s in SKILLS if s in job.lower()]
                matched=[s for s in required if s in resume_skills]
                missing=[s for s in required if s not in resume_skills]
                semantic=float(util.cos_sim(job_embedding,model.encode(clean(text),convert_to_tensor=True)).item()*100)
                skill_score=(len(matched)/len(required)*100) if required else 100
                final=(semantic*0.75)+(skill_score*0.25)
                data.append({
                    "Candidate":f.name.rsplit(".",1)[0],"Email":email(text),"Phone":phone(text),
                    "AI Match (%)":round(final,2),"Semantic Score (%)":round(semantic,2),
                    "Skill Match (%)":round(skill_score,2),"Experience (Years)":experience(text),
                    "Education":education(text),"Matched Skills":", ".join(matched) or "None",
                    "Missing Skills":", ".join(missing) or "None","Resume Text":text
                })
            except Exception as e:
                st.warning(f"Could not process {f.name}: {e}")

    result=pd.DataFrame(data)
    result=result[result["AI Match (%)"]>=threshold].sort_values("AI Match (%)",ascending=False).reset_index(drop=True)
    if result.empty: st.warning("No candidates passed the selected threshold."); st.stop()
    result.insert(0,"Rank",range(1,len(result)+1))
    st.session_state["result"]=result

if "result" in st.session_state:
    df=st.session_state["result"]
    st.divider()
    st.subheader("📊 AI Recruiter Dashboard")
    a,b,c,d=st.columns(4)
    a.metric("Candidates",len(df))
    b.metric("Average AI Match",f"{df['AI Match (%)'].mean():.1f}%")
    c.metric("Top Candidate",df.iloc[0]["Candidate"])
    d.metric("Top Score",f"{df.iloc[0]['AI Match (%)']:.1f}%")

    t1,t2,t3=st.tabs(["🏆 Ranking","🔎 Candidate Analysis","📥 Export"])
    with t1:
        st.dataframe(df[["Rank","Candidate","AI Match (%)","Semantic Score (%)","Skill Match (%)","Experience (Years)","Education","Matched Skills","Missing Skills"]],
                     use_container_width=True,hide_index=True)
        st.bar_chart(df.set_index("Candidate")[["AI Match (%)"]])

    with t2:
        selected=st.selectbox("Choose candidate",df["Candidate"].tolist())
        r=df[df["Candidate"]==selected].iloc[0]
        x,y=st.columns(2)
        with x:
            st.metric("AI Match",f"{r['AI Match (%)']}%")
            st.write("**Email:**",r["Email"])
            st.write("**Phone:**",r["Phone"])
            st.write("**Experience:**",f"{r['Experience (Years)']} years")
            st.write("**Education:**",r["Education"])
        with y:
            st.write("### ✅ Matched Skills")
            st.success(r["Matched Skills"])
            st.write("### ⚠️ Missing Skills")
            st.warning(r["Missing Skills"])
        with st.expander("View extracted resume text"):
            st.text(r["Resume Text"])

    with t3:
        cols=["Rank","Candidate","Email","Phone","AI Match (%)","Semantic Score (%)","Skill Match (%)","Experience (Years)","Education","Matched Skills","Missing Skills"]
        st.download_button("⬇️ Download AI Ranking CSV",df[cols].to_csv(index=False).encode(), "ai_candidate_ranking.csv","text/csv",use_container_width=True)

st.caption("Educational project • AI-assisted screening • Keep human review in the hiring loop.")
