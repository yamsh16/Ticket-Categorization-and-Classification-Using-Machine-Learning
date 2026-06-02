import streamlit as st
from predict import predict_ticket

st.set_page_config(
    page_title="AI Ticket Intelligence",
    page_icon="🎫",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.metric-card {
    background: linear-gradient(135deg,#1e293b,#0f172a);
    padding:20px;
    border-radius:15px;
    border:1px solid #334155;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}

.big-title {
    font-size:42px;
    font-weight:700;
    text-align:center;
    color:white;
}

.sub-title {
    text-align:center;
    color:#94A3B8;
    margin-bottom:30px;
}

.prediction-box {
    background:#111827;
    padding:20px;
    border-radius:12px;
    border-left:5px solid #3B82F6;
}

.rootcause-box {
    background:#111827;
    padding:25px;
    border-radius:12px;
    border-left:5px solid #10B981;
}

.stTextArea textarea {
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class='big-title'>
        🎫 AI Ticket Intelligence Platform
    </div>
    <div class='sub-title'>
        Automatic Type, Queue, Priority & Root Cause Prediction
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        width=120
    )

    st.title("Model Details")

    st.success("Type Classifier")
    st.success("Queue Classifier")
    st.success("Priority Classifier")
    st.success("Root Cause Generator")

    st.divider()

    st.markdown("""
    ### Workflow

    1. Enter ticket
    2. Predict metadata
    3. Generate root cause
    4. Route automatically
    """)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("Ticket Description")

ticket = st.text_area(
    "",
    height=220,
    placeholder="""
Example:

User unable to login after password reset.
VPN access not working.
Receiving authentication failure.
"""
)

# --------------------------------------------------
# BUTTON
# --------------------------------------------------

if st.button(
    "🚀 Analyze Ticket",
    use_container_width=True
):

    if ticket.strip() == "":
        st.warning("Please enter ticket description.")
        st.stop()

    with st.spinner("AI analyzing ticket..."):

        result = predict_ticket(ticket)

    st.success("Analysis Completed")

    st.divider()

    # --------------------------------------------------
    # KPI ROW
    # --------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class='metric-card'>
            <h4>📂 Ticket Type</h4>
            <h2>{result['Type']}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            result["Type Confidence"]/100
        )

        st.caption(
            f"Confidence: {result['Type Confidence']}%"
        )

    with c2:
        st.markdown(
            f"""
            <div class='metric-card'>
            <h4>👥 Queue</h4>
            <h2>{result['Queue']}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            result["Queue Confidence"]/100
        )

        st.caption(
            f"Confidence: {result['Queue Confidence']}%"
        )

    with c3:

        priority = result["Priority"]

        if str(priority).lower() in ["critical","p1","high"]:
            icon = "🔴"

        elif str(priority).lower() in ["medium","p2"]:
            icon = "🟡"

        else:
            icon = "🟢"

        st.markdown(
            f"""
            <div class='metric-card'>
            <h4>{icon} Priority</h4>
            <h2>{priority}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            result["Priority Confidence"]/100
        )

        st.caption(
            f"Confidence: {result['Priority Confidence']}%"
        )

    st.divider()

    # --------------------------------------------------
    # SUMMARY SECTION
    # --------------------------------------------------

    left, right = st.columns([1,1])

    with left:

        st.markdown(
            """
            ### 🎯 Ticket Routing Decision
            """
        )

        st.markdown(
            f"""
            <div class='prediction-box'>
                <b>Type:</b> {result['Type']}<br><br>
                <b>Queue:</b> {result['Queue']}<br><br>
                <b>Priority:</b> {result['Priority']}
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            ### 📊 Confidence Summary
            """
        )

        st.metric(
            "Average Confidence",
            f"{round((result['Type Confidence'] + result['Queue Confidence'] + result['Priority Confidence'])/3,2)}%"
        )

    st.divider()

    # --------------------------------------------------
    # ROOT CAUSE
    # --------------------------------------------------

    st.markdown(
        """
        ## 🤖 AI Root Cause Analysis
        """
    )

    st.markdown(
        f"""
        <div class='rootcause-box'>
        {result['Root Cause & Resolution']}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.download_button(
        label="📄 Export Result",
        data=str(result),
        file_name="ticket_prediction.txt"
    )