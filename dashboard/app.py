import streamlit as st
import pandas as pd
from pathlib import Path

# Page setup
st.set_page_config(
    page_title="CarePulse | ICU Monitoring",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 750;
        color: #14B8A6;
        margin-bottom: 0;
    }
    .app-subtitle {
        color: #94A3B8;
        margin-top: 4px;
        margin-bottom: 20px;
    }
    div[data-testid="stMetric"] {
        background: #171D29;
        border: 1px solid #293548;
        padding: 16px;
        border-radius: 12px;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 650;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Dataset
DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "vitals.csv"
)

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data["Timestamp"] = pd.to_datetime(
        data["Timestamp"], errors="coerce"
    )
    return data

st.sidebar.title("🏥 CarePulse")
st.sidebar.caption("ICU Patient Monitoring")
page = st.sidebar.radio(
    "Navigation",
    [
        "Monitoring Overview",
        "Patient Records",
        "AI Risk Assessment",
        "Federated Learning",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Dataset demonstration — not for clinical decisions.")

st.markdown(
    '<p class="app-title">CarePulse ICU Monitoring</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="app-subtitle">Patient vital signs and risk overview</p>',
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found at: {DATA_PATH}")
    st.info("Check the CSV filename and its location.")
    st.stop()

df = load_data(str(DATA_PATH))

required_columns = [
    "Patient ID",
    "Heart Rate",
    "Respiratory Rate",
    "Oxygen Saturation",
    "Body Temperature",
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Timestamp",
    "Risk Category",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error(f"Required dataset columns are missing: {missing}")
    st.stop()

# Shared helpers
def risk_counts(data):
    risk = data["Risk Category"].astype(str).str.strip().str.lower()
    high = int((risk == "high risk").sum())
    low = int((risk == "low risk").sum())
    return high, low



def show_vitals(record):
    st.markdown("##### Vital Signs")

    c1, c2 = st.columns(2)

    c1.metric(
        "Heart Rate",
        f"{record['Heart Rate']:.0f} bpm",
    )
    c2.metric(
        "Oxygen Saturation",
        f"{record['Oxygen Saturation']:.1f}%",
    )

    c3, c4 = st.columns(2)

    c3.metric(
        "Respiratory Rate",
        f"{record['Respiratory Rate']:.0f} /min",
    )
    c4.metric(
        "Body Temperature",
        f"{record['Body Temperature']:.1f} °C",
    )

def show_patient_details(record):
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Systolic BP",
        f"{record['Systolic Blood Pressure']:.0f}",
        help="Systolic blood pressure in mmHg",
    )
    c2.metric(
        "Diastolic BP",
        f"{record['Diastolic Blood Pressure']:.0f}",
        help="Diastolic blood pressure in mmHg",
    )

    if "Age" in record.index and pd.notna(record["Age"]):
        c3.metric("Age", f"{record['Age']:.0f} years")
    else:
        c3.metric("Age", "N/A")

    st.caption("Blood pressure values are in mmHg.")

# Page 1: Monitoring overview

if page == "Monitoring Overview":

    st.subheader("ICU Patient Monitor")
    st.caption(
        "Explore recorded patient measurements from the project dataset. "
        "This is not a live clinical monitoring feed."
    )

    # Summary metrics
    high, low = risk_counts(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records", f"{len(df):,}")
    c2.metric("Distinct Patient IDs", f"{df['Patient ID'].nunique():,}")
    c3.metric("High-Risk Labels", f"{high:,}")
    c4.metric("Low-Risk Labels", f"{low:,}")

    st.divider()

    # Search and select a record
    st.markdown("### Patient Monitor")

    search_col, risk_col = st.columns([2, 1])

    with search_col:
        patient_search = st.text_input(
            "Search Patient ID",
            placeholder="Enter a Patient ID",
            key="monitor_search",
        )

    with risk_col:
        risk_filter = st.selectbox(
            "Filter by dataset risk label",
            ["All", "High Risk", "Low Risk"],
            key="monitor_risk_filter",
        )

    filtered = df.copy()

    if patient_search.strip():
        filtered = filtered[
            filtered["Patient ID"].astype(str).str.contains(
                patient_search.strip(), case=False, na=False
            )
        ]

    if risk_filter != "All":
        filtered = filtered[
            filtered["Risk Category"].astype(str).str.strip().str.lower()
            == risk_filter.lower()
        ]

    if filtered.empty:
        st.warning("No records match these filters.")
        st.stop()

    # Keep the selection manageable
    filtered = filtered.sort_values(
        "Timestamp", ascending=False, na_position="last"
    ).head(500).reset_index(drop=True)

    # Two-column patient list and details layout
    list_col, detail_col = st.columns([1, 1.7])

    with list_col:
        st.markdown("#### Patient Records")
        st.caption(f"Showing {len(filtered)} matching records")

        selected_index = st.selectbox(
            "Choose a record",
            options=list(range(len(filtered))),
            format_func=lambda i: (
                f"ID {filtered.iloc[i]['Patient ID']} · "
                f"{filtered.iloc[i]['Risk Category']} · "
                f"{filtered.iloc[i]['Timestamp']}"
            ),
            key="monitor_selected_record",
        )

        display_cols = [
            "Patient ID",
            "Heart Rate",
            "Oxygen Saturation",
            "Risk Category",
        ]

        st.dataframe(
            filtered[["Patient ID", "Heart Rate", "Risk Category"]].head(12),
            use_container_width=True,
            hide_index=True,
        )

    with detail_col:
        record = filtered.iloc[selected_index]

        st.markdown(
            f"#### Selected Record · Patient ID {record['Patient ID']}"
        )
        st.caption(f"Recorded at: {record['Timestamp']}")

        risk_text = str(record["Risk Category"]).strip().upper()

        if risk_text == "HIGH RISK":
            st.error(f"Dataset label: {risk_text}")
        elif risk_text == "LOW RISK":
            st.success(f"Dataset label: {risk_text}")
        else:
            st.info(f"Dataset label: {risk_text}")

        # Vital sign cards
        
        show_vitals(record)

        st.markdown("##### Blood Pressure and Demographics")
        show_patient_details(record)

        st.divider()

        # Compare records for this Patient ID only if there are
        # multiple timestamps; don't imply a trend otherwise.
        same_patient = df[
            df["Patient ID"] == record["Patient ID"]
        ].copy()

        same_patient = (
            same_patient.dropna(subset=["Timestamp"])
            .sort_values("Timestamp")
        )

        st.markdown("##### Recorded Measurements Over Time")

        if len(same_patient) >= 2:
            trend_measure = st.selectbox(
                "Choose a measurement",
                [
                    "Heart Rate",
                    "Oxygen Saturation",
                    "Respiratory Rate",
                    "Body Temperature",
                    "Systolic Blood Pressure",
                    "Diastolic Blood Pressure",
                ],
                key="patient_trend_measure",
            )

            trend = same_patient.set_index("Timestamp")[
                [trend_measure]
            ]

            st.line_chart(trend, height=260)

            st.caption(
                "Trend shown only from records sharing this Patient ID. "
                "Confirm the dataset's patient-history structure before "
                "interpreting this as a real clinical timeline."
            )
        else:
            st.info(
                "Only one timestamped record is available for this Patient ID. "
                "A patient trend cannot be displayed."
            )

    st.divider()

    with st.expander("View recent dataset records"):
        recent = (
            df.sort_values("Timestamp", ascending=False)
            if df["Timestamp"].notna().any()
            else df
        )

        st.dataframe(
            recent[
                [
                    "Patient ID",
                    "Timestamp",
                    "Heart Rate",
                    "Oxygen Saturation",
                    "Respiratory Rate",
                    "Risk Category",
                ]
            ].head(20),
            use_container_width=True,
            hide_index=True,
        )

# Page 2: Patient records
elif page == "Patient Records":

    st.subheader("Patient Record Explorer")
    st.write(
        "Search and inspect an individual record from the dataset."
    )

    search_id = st.text_input(
        "Search by Patient ID",
        placeholder="Enter a Patient ID, for example 25",
    )

    records = df.copy()

    if search_id.strip():
        records = records[
            records["Patient ID"].astype(str).str.contains(
                search_id.strip(), case=False, na=False
            )
        ]

    if records.empty:
        st.warning("No records match that Patient ID.")
        st.stop()

    records = records.reset_index(drop=True)

    selected_index = st.selectbox(
        "Select a record",
        options=list(range(len(records))),
        format_func=lambda i: (
            f"Patient ID: {records.iloc[i]['Patient ID']} "
            f"| {records.iloc[i]['Timestamp']}"
        ),
    )

    record = records.iloc[selected_index]

    risk_text = str(record["Risk Category"]).strip().upper()

    if risk_text == "HIGH RISK":
        st.error(f"Dataset risk label: {risk_text}")
    elif risk_text == "LOW RISK":
        st.success(f"Dataset risk label: {risk_text}")
    else:
        st.info(f"Dataset risk label: {risk_text}")

    show_vitals(record)

    st.markdown("#### Additional Measurements")
    show_patient_details(record)

    st.markdown("#### Record Details")

    details = record.to_frame(name="Value")
    details.index.name = "Measurement"
    st.dataframe(details, use_container_width=True)

    st.caption(
        "The displayed risk label comes from the dataset. "
        "It is not a new prediction from the federated model."
    )


# Page 3: AI Risk Assessment
elif page == "AI Risk Assessment":

    import requests

    st.subheader("AI Patient Risk Assessment")
    st.caption(
        "Enter patient measurements to obtain an experimental "
        "prediction from the federated global model."
    )

    st.warning(
        "Academic demonstration only. Predictions are not clinically "
        "validated and must not be used for diagnosis or treatment."
    )

    with st.form("risk_assessment_form"):
        st.markdown("#### Patient Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age (years)", 18, 120, 45)
            gender_label = st.selectbox("Gender", ["Female", "Male"])
            weight = st.number_input("Weight (kg)", 1.0, 300.0, 75.0)
            height = st.number_input("Height (m)", 0.5, 2.5, 1.70)
            derived_hrv = st.number_input(
                "Derived HRV (dataset feature)",
                min_value=0.05,
                max_value=0.15,
                value=0.10,
                step=0.001,
                format="%.3f",
                help="Enter a value using the dataset's Derived_HRV definition."
            )

        with c2:
            hr = st.number_input("Heart Rate (bpm)", 1, 250, 90)
            rr = st.number_input("Respiratory Rate (/min)", 1, 100, 20)
            bt = st.number_input("Body Temperature (°C)", 25.0, 45.0, 37.5)
            spo2 = st.number_input("Oxygen Saturation (%)", 0.0, 100.0, 95.0)

        with c3:
            sbp = st.number_input("Systolic BP (mmHg)", 1, 300, 130)
            dbp = st.number_input("Diastolic BP (mmHg)", 1, 200, 85)

        submitted = st.form_submit_button(
            "Predict Risk",
            use_container_width=True,
        )

    if submitted:
        if height <= 0 or sbp <= dbp:
            st.error(
                "Please check the measurements. Height must be positive "
                "and systolic BP must exceed diastolic BP for this demo."
            )
        else:
            pulse_pressure = sbp - dbp
            bmi = weight / (height ** 2)
            derived_hrv = 0.100

            patient_data = {
                "Heart Rate": hr,
                "Respiratory Rate": rr,
                "Body Temperature": bt,
                "Oxygen Saturation": spo2,
                "Systolic Blood Pressure": sbp,
                "Diastolic Blood Pressure": dbp,
                "Age": age,
                "Gender": 0 if gender_label == "Female" else 1,
                "Weight (kg)": weight,
                "Height (m)": height,
                "Derived_HRV": derived_hrv,
                "Derived_Pulse_Pressure": pulse_pressure,
                "Derived_BMI": bmi,
                "Derived_MAP": dbp + pulse_pressure / 3,
            }

            try:
                st.json(patient_data)
                response = requests.post(
                    "http://127.0.0.1:5000/predict",
                    json=patient_data,
                    timeout=10,
                )

                if response.ok:
                    result = response.json()

                    st.markdown("#### Prediction Result")

                    if result["prediction"] == 1:
                        st.error("Model output: HIGH RISK")
                    else:
                        st.success("Model output: LOW RISK")

                    st.metric(
                        "Model-estimated high-risk probability",
                        f"{result['high_risk_probability']:.1%}",
                    )

                    st.caption(
                        "This is a model score, not a calibrated clinical "
                        "probability or a measure of clinical certainty."
                    )
                else:
                    st.error(f"API error ({response.status_code})")
                    st.code(response.text)

            except requests.exceptions.RequestException as exc:
                st.error(
                    "Could not connect to the prediction API. "
                    "Check that Flask is running on port 5000."
                )
                st.caption(str(exc))

# Page 4: Federated learning
elif page == "Federated Learning":

    st.subheader("Federated Learning")

    st.write(
        "The project simulates collaborative training across "
        "three hospital clients using a FedAvg-style workflow."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Simulated Clients", "3")
    c2.metric("Communication Rounds", "10")
    c3.metric("Aggregation", "FedAvg-style")

    st.divider()

    st.markdown("#### Training Workflow")

    stages = [
        ("01", "Global model", "The server initializes the model."),
        ("02", "Local training", "Each simulated client trains locally."),
        ("03", "Parameter exchange", "Clients send model parameters."),
        ("04", "Aggregation", "The server combines client parameters."),
        ("05", "Global update", "The updated model is used in the next round."),
    ]

    for number, title, description in stages:
        with st.container(border=True):
            a, b = st.columns([0.15, 0.85])
            a.markdown(f"### {number}")
            b.markdown(f"**{title}**")
            b.write(description)

    st.info(
        "This page describes the experiment setup. "
        "We will add evaluation metrics after verifying them against "
        "the saved model and training results."
    )

st.sidebar.divider()
st.sidebar.caption("CarePulse · Academic project demonstration")
