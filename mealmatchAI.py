import streamlit as st
import pandas as pd
from datetime import datetime
import math
import os
import base64
import json
import pywhatkit as whatsapp

# Simple distance approximation (km)
def calculate_distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111.0


def classify_image_with_ai(image_bytes):
    """Classify an image and return a recommended edibility review."""
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        b64 = base64.b64encode(image_bytes).decode()
        prompt = (
            "You are a food-safety assistant. Given an image (base64) of food, decide whether it is likely "
            "safe for humans, safe for common pets (dogs/cats), and safe for other animals/livestock. "
            "Respond ONLY with a single valid JSON object with keys: edible_human (true/false), compost (true/false), "
            "edible_animal (true/false), notes (short string), confidences (object with keys edible_human, compost, edible_animal with numbers 0-1).\n\n"
            f"Image (base64, truncated): {b64[:2000]}"
        )

        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            text = resp.choices[0].message.content
        except Exception:
            resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=300)
            text = resp.choices[0].text

        try:
            return json.loads(text.strip())
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        raise RuntimeError("Failed to parse AI response")
    except Exception:
        return {
            "edible_human": False,
            "compost": False,
            "edible_animal": True,
            "notes": "AI classification unavailable or failed; please verify manually.",
            "confidences": {"edible_human": 0.0, "compost": 0.0, "edible_animal": 0.5},
        }


def initialize_state():
    if 'reports' not in st.session_state:
        st.session_state.reports = [
            {
                "id": 1,
                "restaurant": "Pizza Palace Downtown",
                "address": "XYZ Street, Downtown",
                "city": "Downtown",
                "lat": 23.033863,
                "lon": 72.585022,
                "waste_description": "Fresh bread, pastries & salads - 8 kg",
                "quantity_kg": 8,
                "reported_at": "2026-06-19 10:30",
                "edible_human": True,
                "edible_pet": False,
                "edible_animal": True,
                "compost": False,
                "notes": "All within best-by date, no spoilage. Excellent for food banks.",
                "status": "Available",
                "claimed_by": None,
                "image_b64": None,
                "ai_review": None,
                "admin_approved": False,
                "ai_verified": False,
            },
            {
                "id": 2,
                "restaurant": "Green Cafe North",
                "address": "456 ABC, Northside",
                "city": "Northside",
                "lat": 40.7580,
                "lon": -73.9855,
                "waste_description": "Day-old bread, fruits & vegetable scraps - 12 kg",
                "quantity_kg": 12,
                "reported_at": "2026-06-19 09:15",
                "edible_human": False,
                "edible_pet": False,
                "edible_animal": False,
                "compost": True,
                "notes": "Good for composting",
                "status": "Available",
                "claimed_by": "ABC SHELTER VOLUNTEER",
                "image_b64": None,
                "ai_review": None,
                "admin_approved": False,
                "ai_verified": False,
            },
            {
                "id": 3,
                "restaurant": "WOW Burgers",
                "address": "789 XYZ, Southside",
                "city": "Southside",
                "lat": 40.6892,
                "lon": -74.0445,
                "waste_description": "Plain boiled rice & unsalted meat",
                "quantity_kg": 6,
                "reported_at": "2026-06-18 18:45",
                "edible_human": False,
                "edible_pet": False,
                "edible_animal": True,
                "compost": True,
                "notes": "Meat may be spoiled for humans. Safe for pets (no onions/garlic) and livestock after inspection.",
                "status": "Available",
                "claimed_by": None,
                "image_b64": None,
                "ai_review": None,
                "admin_approved": False,
                "ai_verified": False,
            },
            {
                "id": 4,
                "restaurant": "Sunny Bakery",
                "address": "321 LMNOP, Downtown",
                "city": "Downtown Metro",
                "lat": 40.7300,
                "lon": -74.0100,
                "waste_description": "Assorted pastries & cakes - 5 kg",
                "quantity_kg": 5,
                "reported_at": "2026-06-19 11:00",
                "edible_human": True,
                "edible_pet": False,
                "edible_animal": False,
                "compost": False,
                "notes": "Human-safe. Contains chocolate & xylitol - NOT safe for dogs/cats.",
                "status": "Claimed",
                "claimed_by": "Local Shelter Volunteer",
                "image_b64": None,
                "ai_review": None,
                "admin_approved": False,
                "ai_verified": False,
            },
        ]

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.user_name = None

    if 'user_lat' not in st.session_state:
        st.session_state.user_lat = 40.7128
    if 'user_lon' not in st.session_state:
        st.session_state.user_lon = -74.0060

    if 'tmp_image_b64' not in st.session_state:
        st.session_state.tmp_image_b64 = None
    if 'ai_result' not in st.session_state:
        st.session_state.ai_result = None


def build_sidebar():
    st.sidebar.header("Your Location (Demo)")
    st.session_state.user_lat = st.sidebar.number_input(
        "Latitude", value=st.session_state.user_lat, format="%.4f", key="lat"
    )
    st.session_state.user_lon = st.sidebar.number_input(
        "Longitude", value=st.session_state.user_lon, format="%.4f", key="lon"
    )
    st.sidebar.caption("Change these numbers to simulate moving. Nearest spots & distances will update automatically.")

    if st.session_state.authenticated:
        st.sidebar.markdown(f"**Logged in as:** {st.session_state.user_name}")
        st.sidebar.markdown(f"**Role:** {st.session_state.user_role}")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.session_state.user_name = None
            st.session_state.tmp_image_b64 = None
            st.session_state.ai_result = None
            st.rerun()


def render_login():
    st.title("MealMatch Login")
    st.info("Choose your role and enter your name or organization to continue.")

    name = st.text_input("Name or Organization", key="login_name")
    role = st.selectbox(
        "I am a:",
        ["Volunteer / Shelter", "Admin", "Restaurant / Donor"],
        key="login_role",
    )

    if st.button("Continue"):
        if not name:
            st.warning("Please enter your name or organization.")
        else:
            st.session_state.authenticated = True
            st.session_state.user_name = name
            st.session_state.user_role = role
            st.success(f"Welcome, {name}! Redirecting to your {role} dashboard.")
            st.rerun()


def volunteer_page():
    st.title("Volunteer / Shelter Dashboard")
    st.info("Locate available food donations and claim pickups for your organization.")

    col1, col2 = st.columns(2)
    with col1:
        show_available = st.checkbox("Only available spots", value=True)
    with col2:
        show_human = st.checkbox("Show human-edible only", value=True)

    df = pd.DataFrame(st.session_state.reports)
    if show_available:
        df = df[df["status"] == "Available"]
    if show_human:
        df = df[df["edible_human"] == True]

    if not df.empty:
        df["distance_km"] = df.apply(
            lambda r: calculate_distance(
                st.session_state.user_lat, st.session_state.user_lon, r["lat"], r["lon"]
            ), axis=1,
        )
        df = df.sort_values("distance_km")

    st.subheader("Available Rescue Opportunities")
    if not df.empty:
        map_df = df[["lat", "lon"]].copy()
        map_df.columns = ["latitude", "longitude"]
        st.map(map_df, zoom=11)

        for _, row in df.iterrows():
            with st.container():
                st.markdown(f"### {row['restaurant']}")
                st.caption(f"{row['address']} • ~{row['distance_km']:.1f} km away")
                st.write(f"**Waste:** {row['waste_description']} ({row['quantity_kg']} kg)")
                st.write(f"**Reported:** {row['reported_at']} | **Status:** {row['status']}")
                st.write(
                    "Humans: " + ("✅" if row["edible_human"] else "❌") +
                    " | Animals: " + ("✅" if row["edible_animal"] else "❌") +
                    " | Compost: " + ("✅" if row["compost"] else "❌")
                )

                with st.expander("Safety notes and details"):
                    st.write(row["notes"])
                    if row.get("claimed_by"):
                        st.info(f"Claimed by: {row['claimed_by']}")
                    if row.get("ai_review"):
                        st.write(f"**AI Review:** {row['ai_review']}")

                if row["status"] == "Available":
                    if st.button(f"Claim pickup #{row['id']}", key=f"claim_{row['id']}"):
                        found = False
                        for report in st.session_state.reports:
                            if report["id"] == row["id"]:
                                if report["status"] == "Available":
                                    report["status"] = "Claimed"
                                    report["claimed_by"] = st.session_state.user_name
                                    found = True
                                break
                        if found:
                            st.success("Pickup claimed. The donor will be notified in a real deployment.")
                        else:
                            st.warning("This pickup was already claimed by another volunteer.")
                        st.rerun()
    else:
        st.warning("No opportunities match the selected filters.")

    st.subheader("My Claimed Pickups")
    my_claims = [r for r in st.session_state.reports if r.get("claimed_by") == st.session_state.user_name]
    if my_claims:
        for claim in my_claims:
            st.write(f"- #{claim['id']} {claim['restaurant']} — {claim['waste_description']} ({claim['status']})")
    else:
        st.write("You have not claimed any pickups yet.")


def admin_page():
    st.title("Admin Dashboard")
    st.info("Review reports, verify AI classifications, and update statuses.")

    report_df = pd.DataFrame(st.session_state.reports)
    st.markdown("### Current Reports")
    st.dataframe(report_df[["id", "restaurant", "city", "quantity_kg", "status", "claimed_by", "admin_approved"]])

    for report in st.session_state.reports:
        with st.expander(f"Report #{report['id']} — {report['restaurant']}"):
            st.write(f"**Address:** {report['address']}")
            st.write(f"**City / Area:** {report['city']}")
            st.write(f"**Description:** {report['waste_description']}")
            st.write(f"**Quantity:** {report['quantity_kg']} kg")
            st.write(f"**Status:** {report['status']}")
            st.write(f"**Claimed By:** {report.get('claimed_by') or 'None'}")
            st.write(f"**Admin Approved:** {'✅ Yes' if report.get('admin_approved') else '❌ No'}")
            st.write(
                "Humans: " + ("✅" if report["edible_human"] else "❌") +
                " | Pets: " + ("✅" if report.get("edible_pet", False) else "❌") +
                " | Animals: " + ("✅" if report["edible_animal"] else "❌") +
                " | Compost: " + ("✅" if report["compost"] else "❌")
            )
            st.write(f"**Notes:** {report['notes']}")
            if report.get("ai_review"):
                st.write(f"**AI review:** {report['ai_review']}")

            if report.get("image_b64"):
                st.write("*Image available for review.*")
            else:
                st.write("*No image uploaded for this report.*")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"Mark Available #{report['id']}", key=f"admin_avail_{report['id']}"):
                    report["status"] = "Available"
                    report["claimed_by"] = None
                    st.success("Report marked available.")
                    st.rerun()

            with col2:
                if st.button(f"Mark Claimed #{report['id']}", key=f"admin_claimed_{report['id']}"):
                    report["status"] = "Claimed"
                    if not report.get("claimed_by"):
                        report["claimed_by"] = "Admin assigned"
                    st.success("Report marked claimed.")
                    st.rerun()

            with col3:
                if st.button(f"Approve Report #{report['id']}", key=f"admin_approve_{report['id']}"):
                    report["admin_approved"] = True
                    st.success("Report approved for verification.")
                    st.rerun()

            if report.get("image_b64") and report.get("admin_approved") and not report.get("ai_verified"):
                if st.button(f"AI verify image #{report['id']}", key=f"ai_verify_{report['id']}"):
                    image_bytes = base64.b64decode(report["image_b64"])
                    with st.spinner("Running AI verification..."):
                        ai_result = classify_image_with_ai(image_bytes)
                    report["ai_review"] = json.dumps(ai_result, indent=2)
                    report["ai_verified"] = True
                    st.success("AI verification completed and saved to report.")
                    st.rerun()
            elif report.get("image_b64") and not report.get("admin_approved"):
                st.warning("⚠️ Approve this report first before AI verification.")
            elif report.get("ai_verified"):
                st.info("✅ AI verification already completed for this report.")

            note_update = st.text_area(
                "Update notes", value=report["notes"], key=f"admin_notes_{report['id']}", height=120
            )
            if st.button(f"Save notes #{report['id']}", key=f"save_notes_{report['id']}"):
                report["notes"] = note_update
                st.success("Notes updated.")
                st.rerun()

            st.subheader(f"Edit Classifications for Report #{report['id']}")
            e_human = st.checkbox("Safe for humans", value=report.get("edible_human", False), key=f"edit_human_{report['id']}")
            e_pet = st.checkbox("Safe for pets", value=report.get("edible_pet", False), key=f"edit_pet_{report['id']}")
            e_animal = st.checkbox("Safe for other animals", value=report.get("edible_animal", False), key=f"edit_animal_{report['id']}")
            e_compost = st.checkbox("Safe for composting", value=report.get("compost", False), key=f"edit_compost_{report['id']}")

            if st.button(f"Save classifications #{report['id']}", key=f"save_class_{report['id']}"):
                report["edible_human"] = e_human
                report["edible_pet"] = e_pet
                report["edible_animal"] = e_animal
                report["compost"] = e_compost
                st.success("Classifications updated.")
                st.rerun()


def donor_page():
    st.title("Restaurant / Donor Reporting")
    st.info("Upload surplus pictures and provide quantity and safety details for volunteers.")

    uploaded_file = st.file_uploader("Upload photo of surplus (optional)", type=["png", "jpg", "jpeg"], key="donor_image")
    r_name = st.text_input("Restaurant / Store Name", "Your Restaurant Name", key="donor_name")
    r_address = st.text_input("Full Address", "123 Example Street, Downtown Metro", key="donor_address")
    r_city = st.text_input("City / Area", "Downtown Metro", key="donor_city")
    r_lat = st.number_input("Latitude (approx)", value=40.71, format="%.4f", key="donor_lat")
    r_lon = st.number_input("Longitude (approx)", value=-74.01, format="%.4f", key="donor_lon")
    waste_desc = st.text_area("Describe the surplus/waste", "E.g. Fresh bread, fruits, cooked rice - total 10 kg", key="donor_desc")
    qty = st.number_input("Quantity (kg)", min_value=1, value=5, key="donor_qty")

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded photo", use_column_width=True)
        if st.button("🔍 Analyze Photo with AI", key="donor_analyze"):
            image_bytes = uploaded_file.read()
            st.session_state.tmp_image_b64 = base64.b64encode(image_bytes).decode()
            with st.spinner("Analyzing image (AI)..."):
                st.session_state.ai_result = classify_image_with_ai(image_bytes)
            st.success("AI provided suggested classifications. Please verify them below.")

    ai = st.session_state.get("ai_result")
    st.subheader("Suggested Classification")
    suggested_human = ai.get("edible_human") if ai else True
    suggested_pet = ai.get("compost") if ai else False
    suggested_animal = ai.get("edible_animal") if ai else False

    e_human = st.checkbox("Safe for humans", value=suggested_human, key="donor_human")
    e_pet = st.checkbox("Safe for pets", value=suggested_pet, key="donor_pet")
    e_animal = st.checkbox("Safe for other animals", value=suggested_animal, key="donor_animal")

    safety_notes_default = ai.get("notes") if ai else "Please verify the food condition and ingredients."
    safety_notes = st.text_area("Safety notes / reasons for classification", safety_notes_default, key="donor_notes")
    verified = st.checkbox("I verify this report is accurate", value=False, key="donor_verified")

    if st.button("Submit Report", key="donor_submit"):
        if not verified:
            st.warning("Please verify the report before submitting.")
        else:
            new_id = max([r["id"] for r in st.session_state.reports]) + 1 if st.session_state.reports else 1
            new_report = {
                "id": new_id,
                "restaurant": r_name,
                "address": r_address,
                "city": r_city,
                "lat": r_lat,
                "lon": r_lon,
                "waste_description": waste_desc,
                "quantity_kg": qty,
                "reported_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "edible_human": bool(e_human),
                "edible_pet": bool(e_pet),
                "edible_animal": bool(e_animal),
                "compost": bool(e_pet),
                "notes": safety_notes,
                "status": "Available",
                "claimed_by": None,
                "image_b64": st.session_state.get("tmp_image_b64"),
                "ai_review": None,
                "admin_approved": False,
                "ai_verified": False,
            }
            st.session_state.reports.append(new_report)
            st.success("Report added successfully! Volunteers can now see and claim it.")
            st.balloons()
            st.session_state.tmp_image_b64 = None
            st.session_state.ai_result = None


def main():
    st.set_page_config(
        page_title="MealMatch - Role-based Access",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_state()
    build_sidebar()

    if not st.session_state.authenticated:
        render_login()
        return

    if st.session_state.user_role == "Volunteer / Shelter":
        volunteer_page()
    elif st.session_state.user_role == "Admin":
        admin_page()
    elif st.session_state.user_role == "Restaurant / Donor":
        donor_page()
    else:
        st.error("Unknown role. Please logout and login again.")


if __name__ == "__main__":
    main()




