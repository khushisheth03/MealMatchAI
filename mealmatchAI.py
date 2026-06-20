import base64
import json
import math
import os
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    from google.cloud import vision
except ImportError:
    vision = None


ADMIN_WHATSAPP_NUMBER = os.getenv("ADMIN_WHATSAPP_NUMBER", "whatsapp:+1234567890")


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in kilometers."""
    radius_km = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


def send_whatsapp_message(to_number, message):
    """Send WhatsApp message with Twilio when configured; otherwise use demo mode."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_WHATSAPP_FROM")

    if not account_sid or not auth_token or not from_number:
        return {
            "status": "demo",
            "message": "Twilio is not configured, so no real WhatsApp message was sent.",
        }

    try:
        from twilio.rest import Client

        client = Client(account_sid, auth_token)
        sent = client.messages.create(body=message, from_=from_number, to=to_number)
        return {"status": "sent", "sid": sent.sid}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def analyze_food_safety_human(labels, safe_search):
    """Determine if food is safe for human consumption."""
    unsafe_keywords = [
        "mold",
        "rot",
        "spoiled",
        "expired",
        "contaminated",
        "toxic",
        "poison",
        "waste",
    ]
    safe_keywords = [
        "fresh",
        "bread",
        "fruit",
        "vegetable",
        "salad",
        "pastry",
        "baked",
        "clean",
    ]

    label_lower = " ".join([label.lower() for label in labels])

    for keyword in unsafe_keywords:
        if keyword in label_lower:
            return False

    if vision and safe_search:
        adult_likelihood = safe_search.adult
        if adult_likelihood in (
            vision.Likelihood.LIKELY,
            vision.Likelihood.VERY_LIKELY,
        ):
            return False

    for keyword in safe_keywords:
        if keyword in label_lower:
            return True

    return len(labels) > 0


def analyze_food_safety_animal(labels):
    """Determine if food is safe for animals."""
    unsafe_for_animals = [
        "chocolate",
        "onion",
        "garlic",
        "avocado",
        "grape",
        "xylitol",
        "alcohol",
        "toxic",
    ]
    label_lower = " ".join([label.lower() for label in labels])

    for keyword in unsafe_for_animals:
        if keyword in label_lower:
            return False

    return len(labels) > 0


def analyze_compost_safety(labels):
    """Determine if food can be composted."""
    non_compostable = ["plastic", "metal", "glass", "styrofoam", "container"]
    label_lower = " ".join([label.lower() for label in labels])

    for keyword in non_compostable:
        if keyword in label_lower:
            return False

    return True


def generate_food_notes(labels, safe_search):
    """Generate descriptive notes about the food."""
    if not labels:
        return "Unable to identify food items in the image."

    notes = f"Detected: {', '.join(labels[:5])}. "

    if vision and safe_search and safe_search.adult == vision.Likelihood.POSSIBLE:
        notes += "Contains potentially unsafe content. "

    notes += "Please verify classifications manually if unsure."
    return notes


def classify_image_with_ai(image_bytes):
    """Classify image with Google Vision if available; otherwise return a manual-review fallback."""
    if not vision:
        return {
            "category": "Manual Review",
            "labels": [],
            "edible_human": True,
            "edible_animal": False,
            "compost": False,
            "notes": "Google Vision is not installed. Please verify classifications manually.",
        }

    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        label_response = client.label_detection(image=image)
        safe_response = client.safe_search_detection(image=image)

        labels = [label.description for label in label_response.label_annotations]
        safe_search = safe_response.safe_search_annotation

        return {
            "category": labels[0] if labels else "Unknown",
            "labels": labels,
            "edible_human": analyze_food_safety_human(labels, safe_search),
            "edible_animal": analyze_food_safety_animal(labels),
            "compost": analyze_compost_safety(labels),
            "notes": generate_food_notes(labels, safe_search),
        }
    except Exception as exc:
        return {
            "category": "Manual Review",
            "labels": [],
            "edible_human": True,
            "edible_animal": False,
            "compost": False,
            "notes": f"AI classification failed: {exc}. Please verify manually.",
        }


def initialize_state():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "dashboard"

    if "setup_shown" not in st.session_state:
        st.session_state.setup_shown = False

    if "reports" not in st.session_state:
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
                "edible_animal": True,
                "compost": False,
                "notes": "All within best-by date, no spoilage. Excellent for food banks.",
                "status": "Available",
                "claimed_by": None,
                "claimed_by_phone": None,
                "image_b64": None,
                "ai_review": None,
                "admin_approved": True,
                "ai_verified": True,
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
                "edible_animal": False,
                "compost": True,
                "notes": "Good for composting",
                "status": "Available",
                "claimed_by": "ABC SHELTER VOLUNTEER",
                "claimed_by_phone": None,
                "image_b64": None,
                "ai_review": None,
                "admin_approved": True,
                "ai_verified": True,
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
                "edible_animal": True,
                "compost": False,
                "notes": "Meat may be spoiled for humans. Safe for pets after inspection.",
                "status": "Available",
                "claimed_by": None,
                "claimed_by_phone": None,
                "image_b64": None,
                "ai_review": None,
                "admin_approved": True,
                "ai_verified": True,
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
                "edible_animal": False,
                "compost": False,
                "notes": "Human-safe. Contains chocolate & xylitol - NOT safe for animals.",
                "status": "Claimed",
                "claimed_by": "Local Shelter Volunteer",
                "claimed_by_phone": None,
                "image_b64": None,
                "ai_review": None,
                "admin_approved": True,
                "ai_verified": True,
            },
        ]

    defaults = {
        "authenticated": False,
        "user_role": None,
        "user_name": None,
        "user_lat": 40.7128,
        "user_lon": -74.0060,
        "tmp_image_b64": None,
        "ai_result": None,
        "active_message_form": None,
        "message_form_type": None,
        "message_phone": "",
        "message_text": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_sidebar():
    st.sidebar.header("Your Location (Demo)")
    st.session_state.user_lat = st.sidebar.number_input(
        "Latitude", value=st.session_state.user_lat, format="%.4f", key="lat"
    )
    st.session_state.user_lon = st.sidebar.number_input(
        "Longitude", value=st.session_state.user_lon, format="%.4f", key="lon"
    )
    st.sidebar.caption("Change these numbers to simulate moving.")

    if st.session_state.authenticated:
        st.sidebar.markdown(f"**Logged in as:** {st.session_state.user_name}")
        st.sidebar.markdown(f"**Role:** {st.session_state.user_role}")
        st.sidebar.divider()

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Back", key="back_btn", use_container_width=True):
                logout()
        with col2:
            if st.button("Logout", key="logout_btn", use_container_width=True):
                logout()


def logout():
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.user_name = None
    st.session_state.tmp_image_b64 = None
    st.session_state.ai_result = None
    st.session_state.current_page = "dashboard"
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


def render_safety_badge(row):
    if row["edible_human"]:
        st.success("Human Consumption")
    elif row["edible_animal"]:
        st.info("Animal Feed")
    elif row["compost"]:
        st.warning("Compost")
    else:
        st.error("Manual Review Needed")


def volunteer_page():
    st.title("Volunteer / Shelter Dashboard")
    st.info("Locate available food donations and claim pickups for your organization.")

    show_available = st.checkbox("Only available spots", value=True)
    df = pd.DataFrame(st.session_state.reports)

    df = df[(df["admin_approved"] == True) & (df["ai_verified"] == True)]
    if show_available:
        df = df[df["status"] == "Available"]

    if not df.empty:
        df = df.copy()
        df["distance_km"] = df.apply(
            lambda r: calculate_distance(
                st.session_state.user_lat,
                st.session_state.user_lon,
                r["lat"],
                r["lon"],
            ),
            axis=1,
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
                st.caption(f"{row['address']} - ~{row['distance_km']:.1f} km away")
                render_safety_badge(row)
                st.write(f"**Waste:** {row['waste_description']} ({row['quantity_kg']} kg)")
                st.write(f"**Reported:** {row['reported_at']} | **Status:** {row['status']}")
                st.write(
                    "Humans: "
                    + ("Yes" if row["edible_human"] else "No")
                    + " | Animals: "
                    + ("Yes" if row["edible_animal"] else "No")
                    + " | Compost: "
                    + ("Yes" if row["compost"] else "No")
                )

                with st.expander("Safety notes and details"):
                    st.write(row["notes"])
                    if row.get("claimed_by"):
                        st.info(f"Claimed by: {row['claimed_by']}")
                    if row.get("ai_review"):
                        st.write(f"**AI Review:** {row['ai_review']}")

                if row["status"] == "Available":
                    col_claim1, col_claim2 = st.columns([2, 1])
                    with col_claim1:
                        if st.button(
                            f"Claim pickup #{row['id']}",
                            key=f"claim_{row['id']}",
                            use_container_width=True,
                        ):
                            for report in st.session_state.reports:
                                if report["id"] == row["id"]:
                                    report["status"] = "Claimed"
                                    report["claimed_by"] = st.session_state.user_name
                                    st.success("Pickup claimed!")
                                    st.rerun()

                    with col_claim2:
                        if st.button(
                            "Message",
                            key=f"msg_{row['id']}",
                            use_container_width=True,
                        ):
                            st.session_state.active_message_form = f"volunteer_{row['id']}"
                            st.rerun()

                    if st.session_state.get("active_message_form") == f"volunteer_{row['id']}":
                        render_volunteer_message_form(row)
    else:
        st.warning("No opportunities match the selected filters.")

    render_my_claims()


def render_volunteer_message_form(row):
    st.divider()
    st.subheader("Send WhatsApp Message to Donor")

    contact_phone = st.text_input(
        "Your WhatsApp number",
        placeholder="whatsapp:+1234567890",
        key=f"vol_phone_{row['id']}",
    )
    msg_text = st.text_area(
        "Your message",
        value=(
            f"Hi, I'm interested in claiming the pickup from {row['restaurant']}. "
            "Can I contact you about pickup details?"
        ),
        height=100,
        key=f"vol_text_{row['id']}",
    )

    col_send, col_cancel = st.columns(2)
    with col_send:
        if st.button("Send Message", key=f"send_vol_{row['id']}", use_container_width=True):
            if not contact_phone:
                st.error("Please enter your WhatsApp number")
            else:
                admin_msg = (
                    f"Volunteer Inquiry\n\n"
                    f"{st.session_state.user_name}\n"
                    f"{contact_phone}\n\n"
                    f"{row['restaurant']}\n"
                    f"Report #{row['id']}\n\n"
                    f"Message:\n{msg_text}"
                )
                result = send_whatsapp_message(ADMIN_WHATSAPP_NUMBER, admin_msg)
                handle_message_result(result, "Message sent to admin!")

    with col_cancel:
        if st.button("Cancel", key=f"cancel_vol_{row['id']}", use_container_width=True):
            st.session_state.active_message_form = None
            st.rerun()


def render_my_claims():
    st.subheader("My Claimed Pickups")
    my_claims = [
        report
        for report in st.session_state.reports
        if report.get("claimed_by") == st.session_state.user_name
    ]

    if not my_claims:
        st.write("You have not claimed any pickups yet.")
        return

    for claim in my_claims:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(
                f"- #{claim['id']} {claim['restaurant']} - "
                f"{claim['waste_description']} ({claim['status']})"
            )
        with col2:
            if st.button(
                "Contact",
                key=f"contact_donor_{claim['id']}",
                help="Contact donor",
                use_container_width=True,
            ):
                st.session_state.active_message_form = f"claimed_{claim['id']}"
                st.rerun()

        if st.session_state.get("active_message_form") == f"claimed_{claim['id']}":
            st.divider()
            st.subheader("Send Message to Donor")
            donor_msg = st.text_area(
                "Your message",
                value=(
                    f"Hi, I'm here for the pickup of {claim['waste_description']}. "
                    "When is convenient for pickup?"
                ),
                height=100,
                key=f"claimed_text_{claim['id']}",
            )

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                if st.button(
                    "Send to Admin",
                    key=f"send_claimed_{claim['id']}",
                    use_container_width=True,
                ):
                    admin_msg = (
                        f"Pickup Ready\n\n"
                        f"{st.session_state.user_name} is ready for pickup.\n\n"
                        f"{claim['restaurant']}\n"
                        f"Report #{claim['id']}\n\n"
                        f"Message:\n{donor_msg}"
                    )
                    result = send_whatsapp_message(ADMIN_WHATSAPP_NUMBER, admin_msg)
                    handle_message_result(result, "Message sent to admin to relay to donor!")

            with col_d2:
                if st.button(
                    "Cancel",
                    key=f"cancel_claimed_{claim['id']}",
                    use_container_width=True,
                ):
                    st.session_state.active_message_form = None
                    st.rerun()
            st.divider()


def handle_message_result(result, success_message):
    if result["status"] == "sent":
        st.success(success_message)
        st.session_state.active_message_form = None
        st.rerun()
    elif result["status"] == "demo":
        st.info(f"Demo Mode: {result.get('message')}")
        st.session_state.active_message_form = None
        st.rerun()
    else:
        st.error(f"Failed: {result.get('message')}")


def admin_page():
    st.title("Admin Dashboard")
    st.info("Review reports, approve images, verify AI classifications, and update statuses.")

    st.markdown("### Approval Workflow")
    st.markdown("1. Donor submits food report with photo")
    st.markdown("2. Admin approves image")
    st.markdown("3. Admin runs AI verification")
    st.markdown("4. Report is ready for volunteers to claim")

    report_df = pd.DataFrame(st.session_state.reports)
    st.markdown("### Current Reports")
    st.dataframe(
        report_df[
            [
                "id",
                "restaurant",
                "city",
                "quantity_kg",
                "status",
                "claimed_by",
                "admin_approved",
                "ai_verified",
            ]
        ]
    )

    for report in st.session_state.reports:
        with st.expander(f"Report #{report['id']} - {report['restaurant']}"):
            st.write(f"**Address:** {report['address']}")
            st.write(f"**City / Area:** {report['city']}")
            st.write(f"**Description:** {report['waste_description']}")
            st.write(f"**Quantity:** {report['quantity_kg']} kg")
            st.write(f"**Status:** {report['status']}")
            st.write(f"**Claimed By:** {report.get('claimed_by') or 'None'}")
            st.write(f"**Admin Approved:** {'Yes' if report.get('admin_approved') else 'No'}")
            st.write(f"**AI Verified:** {'Yes' if report.get('ai_verified') else 'No'}")
            st.write(
                "Humans: "
                + ("Yes" if report["edible_human"] else "No")
                + " | Animals: "
                + ("Yes" if report["edible_animal"] else "No")
                + " | Compost: "
                + ("Yes" if report["compost"] else "No")
            )
            st.write(f"**Notes:** {report['notes']}")

            if report.get("ai_review"):
                st.write(f"**AI review:** {report['ai_review']}")

            if report.get("image_b64"):
                st.image(base64.b64decode(report["image_b64"]), width=400)
            else:
                st.write("*No image uploaded for this report.*")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(
                    f"Mark Available #{report['id']}",
                    key=f"admin_avail_{report['id']}",
                ):
                    report["status"] = "Available"
                    report["claimed_by"] = None
                    st.success("Report marked available.")
                    st.rerun()

            with col2:
                if st.button(
                    f"Mark Claimed #{report['id']}",
                    key=f"admin_claimed_{report['id']}",
                ):
                    report["status"] = "Claimed"
                    if not report.get("claimed_by"):
                        report["claimed_by"] = "Admin assigned"
                    st.success("Report marked claimed.")
                    st.rerun()

            with col3:
                if st.button(f"Approve #{report['id']}", key=f"approve_{report['id']}"):
                    report["admin_approved"] = True
                    report["status"] = "Available"
                    st.success("Approved and published.")
                    st.rerun()

            if report.get("image_b64") and report.get("admin_approved") and not report.get("ai_verified"):
                if st.button(f"AI verify image #{report['id']}", key=f"ai_verify_{report['id']}"):
                    image_bytes = base64.b64decode(report["image_b64"])
                    with st.spinner("Running AI verification..."):
                        ai_result = classify_image_with_ai(image_bytes)
                    report["edible_human"] = bool(ai_result.get("edible_human"))
                    report["edible_animal"] = bool(ai_result.get("edible_animal"))
                    report["compost"] = bool(ai_result.get("compost"))
                    report["notes"] = ai_result.get("notes", report["notes"])
                    report["ai_review"] = json.dumps(ai_result, indent=2)
                    report["ai_verified"] = True
                    msg = (
                        f"AI Verification Complete\n\n"
                        f"Report #{report['id']} - {report['restaurant']}\n\n"
                        f"Humans: {'Yes' if ai_result.get('edible_human') else 'No'}\n"
                        f"Animals: {'Yes' if ai_result.get('edible_animal') else 'No'}\n"
                        f"Compost: {'Yes' if ai_result.get('compost') else 'No'}"
                    )
                    result = send_whatsapp_message(ADMIN_WHATSAPP_NUMBER, msg)
                    if result["status"] == "error":
                        st.warning(
                            "AI verification completed but WhatsApp notification failed: "
                            f"{result.get('message', 'Unknown error')}"
                        )
                    else:
                        st.success("AI verification completed. Admin notified.")
                    st.rerun()
            elif report.get("image_b64") and not report.get("admin_approved"):
                st.warning("Approve this report first before AI verification.")
            elif report.get("ai_verified"):
                st.info("AI verification already completed for this report.")

            note_update = st.text_area(
                "Update notes",
                value=report["notes"],
                key=f"admin_notes_{report['id']}",
                height=120,
            )
            if st.button(f"Save notes #{report['id']}", key=f"save_notes_{report['id']}"):
                report["notes"] = note_update
                st.success("Notes updated.")
                st.rerun()

            st.subheader(f"Edit Classifications for Report #{report['id']}")
            e_human = st.checkbox(
                "Safe for humans",
                value=report.get("edible_human", False),
                key=f"edit_human_{report['id']}",
            )
            e_animal = st.checkbox(
                "Safe for animals",
                value=report.get("edible_animal", False),
                key=f"edit_animal_{report['id']}",
            )
            e_compost = st.checkbox(
                "Safe for composting",
                value=report.get("compost", False),
                key=f"edit_compost_{report['id']}",
            )

            if st.button(f"Save classifications #{report['id']}", key=f"save_class_{report['id']}"):
                report["edible_human"] = e_human
                report["edible_animal"] = e_animal
                report["compost"] = e_compost
                st.success("Classifications updated.")
                st.rerun()


def donor_page():
    st.title("Restaurant / Donor Reporting")
    st.info(
        "Upload surplus pictures and provide quantity and safety details for volunteers. "
        "Admin will verify classifications."
    )

    uploaded_file = st.file_uploader(
        "Upload surplus food photo",
        type=["png", "jpg", "jpeg"],
        key="donor_upload",
    )

    r_name = st.text_input("Restaurant / Store Name", "Your Restaurant Name", key="donor_name")
    r_address = st.text_input("Full Address", "123 Example Street, Downtown Metro", key="donor_address")
    r_city = st.text_input("City / Area", "Downtown Metro", key="donor_city")
    r_lat = st.number_input("Latitude (approx)", value=40.71, format="%.4f", key="donor_lat")
    r_lon = st.number_input("Longitude (approx)", value=-74.01, format="%.4f", key="donor_lon")
    waste_desc = st.text_area(
        "Describe the surplus/waste",
        "E.g. Fresh bread, fruits, cooked rice - total 10 kg",
        key="donor_desc",
    )
    qty = st.number_input("Quantity (kg)", min_value=1, value=5, key="donor_qty")

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded photo", use_container_width=True)
        if st.button("Analyze Photo with AI", key="donor_analyze"):
            image_bytes = uploaded_file.getvalue()
            st.session_state.tmp_image_b64 = base64.b64encode(image_bytes).decode()
            with st.spinner("Analyzing image (AI)..."):
                st.session_state.ai_result = classify_image_with_ai(image_bytes)
            st.success("AI provided suggested classifications. Please verify them below.")
    else:
        st.error("Photo upload is required.")

    ai = st.session_state.get("ai_result")
    st.subheader("Suggested Classification")
    suggested_human = ai.get("edible_human") if ai else True
    suggested_compost = ai.get("compost") if ai else False
    suggested_animal = ai.get("edible_animal") if ai else False

    e_human = st.checkbox("Safe for humans", value=suggested_human, key="donor_human")
    e_animal = st.checkbox("Safe for animals", value=suggested_animal, key="donor_animal")
    e_compost = st.checkbox("Safe for composting", value=suggested_compost, key="donor_compost")

    safety_notes_default = ai.get("notes") if ai else "Please verify the food condition and ingredients."
    safety_notes = st.text_area(
        "Safety notes / reasons for classification",
        safety_notes_default,
        key="donor_notes",
    )
    verified = st.checkbox("I verify this report is accurate", value=False, key="donor_verified")

    if st.button("Submit Report", key="donor_submit"):
        if not uploaded_file:
            st.error("Photo upload is required.")
        elif not verified:
            st.warning("Please verify the report before submitting.")
        else:
            if not st.session_state.tmp_image_b64:
                image_bytes = uploaded_file.getvalue()
                st.session_state.tmp_image_b64 = base64.b64encode(image_bytes).decode()

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
                "edible_animal": bool(e_animal),
                "compost": bool(e_compost),
                "notes": safety_notes,
                "status": "Pending Approval",
                "claimed_by": None,
                "claimed_by_phone": None,
                "image_b64": st.session_state.get("tmp_image_b64"),
                "ai_review": json.dumps(ai, indent=2) if ai else None,
                "admin_approved": False,
                "ai_verified": bool(ai),
            }
            st.session_state.reports.append(new_report)

            msg = f"""
APPROVAL REQUIRED

Report ID: {new_id}

Restaurant:
{r_name}

Category:
{ai['category'] if ai else 'Unknown'}

Quantity:
{qty} kg

Please review and approve.
"""
            result = send_whatsapp_message(ADMIN_WHATSAPP_NUMBER, msg)
            if result["status"] == "error":
                st.warning(
                    "Report submitted but WhatsApp notification failed: "
                    f"{result.get('message', 'Unknown error')}"
                )
            else:
                st.success("Report added successfully! Admin & volunteers notified.")
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
