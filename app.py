import streamlit as st, joblib, pandas as pd

art = joblib.load("model.joblib")
model, FEATURES, ranges = art["model"], art["features"], art["ranges"]
LABELS = ["Low", "Medium", "High"]

# Rules: tips shown when the value is on the unhealthy end of the scale
HIGH_IS_BAD = {
    "study_load": "Break study into 25-min focus blocks with breaks, and talk to your teachers if the workload feels unmanageable.",
    "peer_pressure": "Set boundaries with peers and spend time with people who support you.",
    "future_career_concerns": "Make a small 3-month plan (skills, projects, applications) to turn worry into action.",
    "noise_level": "Find a quieter study spot or use noise-cancelling headphones.",
    "bullying": "Please talk to a trusted person or your institution's student support. You don't have to handle this alone.",
}
LOW_IS_BAD = {
    "sleep_quality": "Keep a fixed sleep schedule and avoid screens 30-60 minutes before bed.",
    "social_support": "Reach out to a friend or family member, and consider joining a student club.",
    "self_esteem": "Track small daily wins and consider speaking with a counsellor.",
    "extracurricular_activities": "Add one enjoyable activity, such as a sport, music, or a hobby group.",
    "living_conditions": "Try to improve your study space, even in small ways (light, tidiness, seating).",
}

st.set_page_config(page_title="Student Wellness Predictor", page_icon="🧠")
st.title("🧠 Student Burnout & Wellness Predictor")
st.caption("Informational tool only, not a medical diagnosis. If you're struggling, please talk to a counsellor or doctor.")

inputs = {}
cols = st.columns(2)
for i, f in enumerate(FEATURES):
    lo, hi = ranges[f]
    with cols[i % 2]:
        inputs[f] = st.slider(f.replace("_", " ").title(), lo, hi, (lo + hi) // 2)

if st.button("Check my stress level"):
    X = pd.DataFrame([inputs])[FEATURES]
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    colors = ["🟢", "🟡", "🔴"]
    st.subheader(f"{colors[pred]} Predicted stress level: {LABELS[pred]}")
    prob_df = pd.DataFrame({"Level": LABELS, "Probability": proba})
    st.bar_chart(prob_df, x="Level", y="Probability", sort=False)

    tips = []
    for f, tip in HIGH_IS_BAD.items():
        lo, hi = ranges[f]
        if inputs[f] >= lo + 0.7 * (hi - lo):
            tips.append(tip)
    for f, tip in LOW_IS_BAD.items():
        lo, hi = ranges[f]
        if inputs[f] <= lo + 0.3 * (hi - lo):
            tips.append(tip)

    st.subheader("Suggested next steps")
    if tips:
        for t in tips: st.write("• " + t)
    else:
        st.write("Your habits look balanced. Keep it up!")
    if pred == 2:
        st.warning("Your predicted stress is high. Consider speaking with a counsellor or someone you trust.")