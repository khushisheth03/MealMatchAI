import base64
import html
import json
import math
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

HF_MODEL = os.getenv("HF_MODEL", "nateraw/food")

LOCATION_PRESETS = {
    "Downtown Metro": (40.7128, -74.0060),
    "Downtown": (23.0339, 72.5850),
    "Northside": (40.7580, -73.9855),
    "Southside": (40.6892, -74.0445),
    "West End": (40.7306, -73.9971),
}


def apply_app_theme():
    st.markdown(
        """
        <style>
        :root {
            --meal-olive: #5f6f38;
            --meal-olive-dark: #34451f;
            --meal-olive-soft: #eef3e2;
            --meal-orange: #e9822e;
            --meal-orange-soft: #fff1e3;
            --meal-ink: #26311c;
        }
        html, body, [data-testid="stAppViewContainer"] {
            color: var(--meal-ink);
        }
        .main .block-container {
            padding-top: 2rem;
            max-width: 1180px;
        }
        h1, h2, h3 {
            color: var(--meal-olive-dark);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--meal-olive-soft), #ffffff);
            border-right: 1px solid #d8dfc6;
        }
        div[data-testid="stMetric"] {
            background: var(--meal-olive-soft);
            border: 1px solid #d8dfc6;
            border-radius: 8px;
            padding: 14px 16px;
        }
        div[data-testid="stExpander"] {
            border-radius: 8px;
            border-color: #d8dfc6;
        }
        .stButton > button {
            border-radius: 7px;
            font-weight: 600;
            border-color: var(--meal-olive);
            color: var(--meal-olive-dark);
        }
        .stButton > button:hover {
            border-color: var(--meal-orange);
            color: var(--meal-orange);
        }
        div[data-testid="stAlert"] {
            border-radius: 8px;
        }
        .meal-chat-message {
            background: #fffaf4;
            border: 1px solid #f1c89f;
            border-left: 4px solid var(--meal-orange);
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 10px;
        }
        .meal-chat-meta {
            color: #65714a;
            font-size: 0.82rem;
            margin-bottom: 4px;
        }
        st.image("Logo(3).png", width=300)
        </style>
        """,
        unsafe_allow_html=True,
        
    )


def get_location_coords(area):
    return LOCATION_PRESETS.get(area, LOCATION_PRESETS["Downtown Metro"])


def render_hf_troubleshooting():
    with st.expander("Hugging Face Connection Help", expanded=False):
        st.write(
            "The DNS error means this computer or hosting environment cannot resolve "
            "`api-inference.huggingface.co`. MealMatch will keep working with offline "
            "safety review, but use these steps to restore the live API:"
        )
        st.markdown(
            """
            1. Open a browser on the same device and visit `https://api-inference.huggingface.co`.
            2. If it does not open, switch networks or disable a VPN/proxy/firewall that blocks Hugging Face.
            3. Confirm your internet DNS works by trying another network, mobile hotspot, or public DNS.
            4. Add your Hugging Face token as `HF_TOKEN` in Streamlit secrets or environment variables.
            5. Restart Streamlit after changing network or token settings.
            6. Try the photo analysis again. If it still fails, continue the demo with offline review and admin approval.
            """
        )


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


def add_chat_message(report_id, sender, role, message):
    st.session_state.chat_messages.append(
        {
            "report_id": report_id,
            "sender": sender,
            "role": role,
            "message": message,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    )


def render_report_chat(report_id, default_message, key_prefix):
    messages = [
        message
        for message in st.session_state.chat_messages
        if message["report_id"] == report_id
    ]

    if messages:
        st.write("Conversation")
        for message in messages:
            safe_meta = html.escape(
                f"{message['created_at']} - {message['sender']} ({message['role']})"
            )
            safe_message = html.escape(message["message"])
            st.markdown(
                (
                    "<div class='meal-chat-message'>"
                    f"<div class='meal-chat-meta'>{safe_meta}</div>"
                    f"<div>{safe_message}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
    else:
        st.info("No messages yet. Start the coordination thread below.")

    quick_messages = [
        default_message,
        "Pickup is confirmed. I will update the report if anything changes.",
        "Can you share the best pickup time and contact person?",
        "Food photo and details reviewed. Please wait for admin approval.",
    ]
    selected_quick_message = st.selectbox(
        "Quick message",
        quick_messages,
        key=f"{key_prefix}_quick_chat_{report_id}",
    )
    text_key = f"{key_prefix}_chat_text_{report_id}"
    if text_key not in st.session_state:
        st.session_state[text_key] = default_message
    if st.button("Use template", key=f"{key_prefix}_use_template_{report_id}"):
        st.session_state[text_key] = selected_quick_message
    new_message = st.text_area(
        "Message",
        height=100,
        key=text_key,
    )
    col_send, col_cancel = st.columns(2)
    with col_send:
        if st.button("Send", key=f"{key_prefix}_send_chat_{report_id}", use_container_width=True):
            if not new_message.strip():
                st.error("Please write a message first.")
            else:
                add_chat_message(
                    report_id,
                    st.session_state.user_name,
                    st.session_state.user_role,
                    new_message.strip(),
                )
                st.success("Message added to the report chat.")
                st.session_state.active_message_form = None
                st.rerun()
    with col_cancel:
        if st.button("Cancel", key=f"{key_prefix}_cancel_chat_{report_id}", use_container_width=True):
            st.session_state.active_message_form = None
            st.rerun()


def get_visible_chat_reports():
    role = st.session_state.user_role
    name = st.session_state.user_name

    if role == "Admin":
        return st.session_state.reports

    if role == "Restaurant / Donor":
        return [
            report
            for report in st.session_state.reports
            if report.get("restaurant") == name
        ]

    if role == "Volunteer / Shelter":
        return [
            report
            for report in st.session_state.reports
            if report.get("claimed_by") == name
            or (
                report.get("admin_approved")
                and report.get("status") == "Available"
            )
        ]

    return []


def render_message_center(key_prefix):
    visible_reports = get_visible_chat_reports()
    report_lookup = {report["id"]: report for report in visible_reports}

    with st.expander("Message Center", expanded=False):
        if not visible_reports:
            st.write("No report conversations are available yet.")
            return

        options = [
            f"#{report['id']} - {report['restaurant']} - {report['status']}"
            for report in visible_reports
        ]
        selected = st.selectbox(
            "Report conversation",
            options,
            key=f"{key_prefix}_message_report",
        )
        selected_id = int(selected.split(" - ")[0].replace("#", ""))
        selected_report = report_lookup[selected_id]

        default_message = (
            f"Hi, this is {st.session_state.user_name}. "
            f"I have an update about report #{selected_id} from {selected_report['restaurant']}."
        )
        render_report_chat(selected_id, default_message, f"{key_prefix}_center")


def analyze_food_safety_human(labels):
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


def generate_food_notes(labels, predictions):
    """Generate descriptive notes about the food."""
    if not labels:
        return "Unable to identify food items in the image."

    notes = f"Detected: {', '.join(labels[:5])}. "

    if predictions:
        top = predictions[0]
        score = top.get("score", 0) * 100
        notes += f"Top model confidence: {score:.1f}%. "

    notes += "Hugging Face results are suggestions; admin should still inspect the photo."
    return notes


def manual_review_result(reason):
    """Create a safe fallback result when AI classification cannot be trusted."""
    return {
        "category": "Manual Review",
        "labels": [],
        "predictions": [],
        "edible_human": False,
        "edible_animal": False,
        "compost": False,
        "notes": reason,
    }


def offline_food_review(description, reason):
    """Fallback classifier for demos when Hugging Face cannot be reached."""
    text = (description or "").lower()

    compost_keywords = [
        "rotten",
        "spoiled",
        "mold",
        "mould",
        "expired",
        "scrap",
        "peel",
        "waste",
        "stale",
    ]
    human_keywords = [
        "fresh",
        "bread",
        "rice",
        "fruit",
        "vegetable",
        "salad",
        "pastry",
        "baked",
        "cooked",
        "packed",
        "sealed",
    ]
    animal_keywords = [
        "plain",
        "rice",
        "unsalted",
        "meat",
        "vegetable",
        "bread",
    ]
    animal_unsafe_keywords = [
        "chocolate",
        "onion",
        "garlic",
        "avocado",
        "grape",
        "xylitol",
        "alcohol",
    ]

    labels = [
        keyword
        for keyword in human_keywords + animal_keywords + compost_keywords
        if keyword in text
    ]
    labels = list(dict.fromkeys(labels)) or ["food item"]

    compost = any(keyword in text for keyword in compost_keywords)
    animal_unsafe = any(keyword in text for keyword in animal_unsafe_keywords)
    edible_human = any(keyword in text for keyword in human_keywords) and not compost
    edible_animal = (
        any(keyword in text for keyword in animal_keywords)
        and not animal_unsafe
        and not compost
    )

    if compost:
        category = "Compost Candidate"
    elif edible_human:
        category = "Human Food Candidate"
    elif edible_animal:
        category = "Animal Feed Candidate"
    else:
        category = "Admin Review Candidate"

    return {
        "category": category,
        "labels": labels,
        "predictions": [
            {"label": label, "score": 0.55}
            for label in labels[:5]
        ],
        "edible_human": edible_human,
        "edible_animal": edible_animal,
        "compost": compost,
        "notes": (
            "Offline safety review used because Hugging Face was unreachable. "
            f"{reason} Admin must inspect the photo before approval."
        ),
        "source": "offline_fallback",
    }


def generate_pickup_brief(description, quantity_kg, area):
    text = (description or "").lower()
    if any(word in text for word in ["rotten", "spoiled", "mold", "mould", "scrap", "peel"]):
        route = "Compost partner"
        checks = "Seal separately, avoid leaking bags, and keep away from edible donations."
    elif any(word in text for word in ["chocolate", "onion", "garlic", "grape", "xylitol"]):
        route = "Human shelter review"
        checks = "Not suitable for animals. Admin should check ingredients and freshness."
    elif any(word in text for word in ["plain", "unsalted", "rice", "meat"]):
        route = "Animal shelter or human shelter review"
        checks = "Confirm freshness, temperature, and no unsafe seasoning."
    else:
        route = "Human shelter review"
        checks = "Confirm packaging, smell, visible spoilage, and best-by timing."

    return (
        f"Suggested route: {route}. Quantity: {quantity_kg} kg. "
        f"Pickup area: {area}. Admin checks: {checks}"
    )


def normalize_hf_predictions(payload):
    """Handle common Hugging Face response shapes for image classification."""
    if isinstance(payload, dict):
        if payload.get("error"):
            if payload.get("estimated_time"):
                seconds = int(payload["estimated_time"])
                raise RuntimeError(f"Model is waking up. Please retry in about {seconds} seconds.")
            raise RuntimeError(str(payload["error"]))

        if isinstance(payload.get("labels"), list):
            return payload["labels"]

        if isinstance(payload.get("predictions"), list):
            return payload["predictions"]

        raise RuntimeError("Hugging Face returned an unsupported response format.")

    if isinstance(payload, list):
        if payload and isinstance(payload[0], list):
            payload = payload[0]

        if all(isinstance(item, dict) for item in payload):
            return payload

    raise RuntimeError("Hugging Face returned no usable image labels.")


def classify_image_with_ai(image_bytes, description=""):
    """Classify an image with Hugging Face Inference API."""
    try:
        try:
            token = st.secrets.get("HF_TOKEN", "")
        except Exception:
            token = ""
        token = token or os.getenv("HF_TOKEN", "")
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/octet-stream",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            data=image_bytes,
            timeout=60,
        )
        
        if response.status_code == 503:
            try:
                payload = response.json()
                normalize_hf_predictions(payload)
                return manual_review_result("Hugging Face model is temporarily unavailable. Please retry.")
            except Exception as exc:
                return manual_review_result(str(exc))

        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError:
            return manual_review_result("Hugging Face returned a non-JSON response.")

        predictions = normalize_hf_predictions(payload)

        labels = [item.get("label", "") for item in predictions if item.get("label")]
        if not labels:
            return manual_review_result("Hugging Face returned predictions without labels.")

        return {
            "category": labels[0].replace("_", " ").title() if labels else "Unknown",
            "labels": labels,
            "predictions": predictions[:5],
            "edible_human": analyze_food_safety_human(labels),
            "edible_animal": analyze_food_safety_animal(labels),
            "compost": analyze_compost_safety(labels),
            "notes": generate_food_notes(labels, predictions),
        }

    except requests.exceptions.HTTPError as e:
        st.error(f"Hugging Face API HTTP Error: {e}")
        return manual_review_result(f"Hugging Face API HTTP error: {e}")
    except requests.exceptions.RequestException as e:
        reason = "Hugging Face is unreachable from this network right now."
        st.warning(reason)
        return offline_food_review(description, reason)
    except Exception as e:
        st.error(f"Image Classification Failed: {e}")
        return manual_review_result(f"Image classification failed: {e}")


def ai_verification_passed(ai_result):
    """Return True only when the model returned a real classification."""
    if not ai_result:
        return False
    return ai_result.get("category") != "Manual Review" and bool(ai_result.get("labels"))


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
                "status": "Pending Image Upload",
                "claimed_by": None,
                "claimed_by_phone": None,
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
                "edible_animal": False,
                "compost": True,
                "notes": "Good for composting",
                "status": "Pending Image Upload",
                "claimed_by": "ABC SHELTER VOLUNTEER",
                "claimed_by_phone": None,
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
                "edible_animal": True,
                "compost": False,
                "notes": "Meat may be spoiled for humans. Safe for pets after inspection.",
                "status": "Pending Image Upload",
                "claimed_by": None,
                "claimed_by_phone": None,
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
                "edible_animal": False,
                "compost": False,
                "notes": "Human-safe. Contains chocolate & xylitol - NOT safe for animals.",
                "status": "Pending Image Upload",
                "claimed_by": "Local Shelter Volunteer",
                "claimed_by_phone": None,
                "image_b64": None,
                "ai_review": None,
                "admin_approved": False,
                "ai_verified": False,
            },
        ]

    defaults = {
        "authenticated": False,
        "user_role": None,
        "user_name": None,
        "user_area": "Downtown Metro",
        "user_lat": 40.7128,
        "user_lon": -74.0060,
        "tmp_image_b64": None,
        "ai_result": None,
        "active_message_form": None,
        "message_form_type": None,
        "message_phone": "",
        "message_text": "",
        "chat_messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_sidebar():
    if st.session_state.authenticated:
        st.sidebar.header("MealMatch")
        st.sidebar.caption("Food rescue coordination")
        st.sidebar.divider()
        st.sidebar.markdown(f"**Signed in:** {st.session_state.user_name}")
        st.sidebar.markdown(f"**Role:** {st.session_state.user_role}")
        st.sidebar.divider()

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
    st.title("MealMatch")
    st.caption("Match surplus food with shelters, animal care groups, and compost partners.")

    col_intro, col_login = st.columns([1.1, 1])
    with col_intro:
        st.subheader("Rescue the right food faster")
        st.write("Restaurants upload a photo, MealMatch suggests safe routing, admins approve, and volunteers coordinate pickup in one place.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Routes", "3", "Human, animal, compost")
        m2.metric("Approval", "Admin", "Required")
        m3.metric("Messages", "Live", "Per report")

    with col_login:
        st.subheader("Sign in")
        name = st.text_input("Name or Organization", key="login_name")
        role = st.selectbox(
            "Role",
            ["Volunteer / Shelter", "Admin", "Restaurant / Donor"],
            key="login_role",
        )

        if st.button("Continue", use_container_width=True):
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
    st.caption("Find approved pickups near your service area and coordinate with donors.")
    render_message_center("volunteer")

    col_area, col_filter = st.columns([2, 1])
    with col_area:
        areas = list(LOCATION_PRESETS.keys())
        selected_index = areas.index(st.session_state.user_area) if st.session_state.user_area in areas else 0
        selected_area = st.selectbox(
            "Service area",
            areas,
            index=selected_index,
            key="volunteer_area",
        )
        st.session_state.user_area = selected_area
        st.session_state.user_lat, st.session_state.user_lon = get_location_coords(selected_area)
    with col_filter:
        show_available = st.checkbox("Show only available pickups", value=True)

    df = pd.DataFrame(st.session_state.reports)

    df = df[df["admin_approved"] == True]
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

    total_available = len(df) if not df.empty else 0
    m1, m2, m3 = st.columns(3)
    m1.metric("Matching pickups", total_available)
    m2.metric("Your area", st.session_state.user_area)
    m3.metric("Visibility", "Approved only")

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
    st.subheader("Report Chat")
    render_report_chat(
        row["id"],
        (
            f"Hi, I'm interested in the pickup from {row['restaurant']}. "
            "Can we coordinate pickup details here?"
        ),
        "volunteer",
    )


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
            st.subheader("Report Chat")
            render_report_chat(
                claim["id"],
                (
                    f"Hi, I'm here for the pickup of {claim['waste_description']}. "
                    "When is convenient for pickup?"
                ),
                "claimed",
            )
            st.divider()


def admin_page():
    st.title("Admin Dashboard")
    st.caption("Review uploads, inspect AI or offline safety notes, and publish safe pickups.")
    render_message_center("admin")
    render_hf_troubleshooting()

    report_df = pd.DataFrame(st.session_state.reports)
    report_df["ai_assistance"] = report_df["ai_verified"].apply(
        lambda value: "Used" if value else "Optional / skipped"
    )
    pending_count = int((report_df["status"] == "Pending Admin Approval").sum())
    available_count = int((report_df["status"] == "Available").sum())
    claimed_count = int((report_df["status"] == "Claimed").sum())
    col_pending, col_available, col_claimed = st.columns(3)
    col_pending.metric("Needs review", pending_count)
    col_available.metric("Available", available_count)
    col_claimed.metric("Claimed", claimed_count)

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
                "ai_assistance",
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
            st.write(f"**AI Assistance:** {'Used' if report.get('ai_verified') else 'Not used / optional'}")
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
                st.subheader("Uploaded Food Image")
                st.image(base64.b64decode(report["image_b64"]), width=400)
            else:
                st.error("No image uploaded for this report. It cannot be approved.")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(
                    f"Mark Available #{report['id']}",
                    key=f"admin_avail_{report['id']}",
                ):
                    if report.get("image_b64") and report.get("admin_approved"):
                        report["status"] = "Available"
                        report["claimed_by"] = None
                        st.success("Report marked available.")
                        st.rerun()
                    else:
                        st.warning("Only image uploads approved by admin can be marked available.")

            with col2:
                if st.button(
                    f"Mark Claimed #{report['id']}",
                    key=f"admin_claimed_{report['id']}",
                ):
                    if report.get("image_b64") and report.get("admin_approved"):
                        report["status"] = "Claimed"
                        if not report.get("claimed_by"):
                            report["claimed_by"] = "Admin assigned"
                        st.success("Report marked claimed.")
                        st.rerun()
                    else:
                        st.warning("Only image uploads approved by admin can be marked claimed.")

            with col3:
                if st.button(f"Approve uploaded image #{report['id']}", key=f"approve_{report['id']}"):
                    if not report.get("image_b64"):
                        st.error("Upload image is missing, so this report cannot be approved.")
                    else:
                        report["admin_approved"] = True
                        report["status"] = "Available"
                        st.success("Uploaded image approved and published for volunteers.")
                        st.rerun()

            if report.get("image_b64") and not report.get("admin_approved"):
                if report.get("ai_verified"):
                    st.info("AI assistance is available. Please inspect the uploaded image before approving.")
                else:
                    st.info("AI was skipped or unavailable. Admin can still inspect the photo and approve.")
            elif report.get("admin_approved"):
                st.success("Admin approval complete.")

            with st.expander("Report Chat"):
                render_report_chat(
                    report["id"],
                    f"Admin note for report #{report['id']}:",
                    "admin",
                )

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
    st.caption(
        "Upload surplus pictures and provide quantity and safety details. "
        "AI suggestions are optional; admin approval is required before publishing."
    )
    render_message_center("donor")
    render_hf_troubleshooting()

    uploaded_file = st.file_uploader(
        "Upload surplus food photo",
        type=["png", "jpg", "jpeg"],
        key="donor_upload",
    )

    r_name = st.text_input("Restaurant / Store Name", "Your Restaurant Name", key="donor_name")
    r_address = st.text_input("Full Address", "123 Example Street, Downtown Metro", key="donor_address")
    r_city = st.selectbox(
        "Pickup Area",
        list(LOCATION_PRESETS.keys()),
        key="donor_city",
    )
    r_lat, r_lon = get_location_coords(r_city)
    waste_desc = st.text_area(
        "Describe the surplus/waste",
        "E.g. Fresh bread, fruits, cooked rice - total 10 kg",
        key="donor_desc",
    )
    qty = st.number_input("Quantity (kg)", min_value=1, value=5, key="donor_qty")
    pickup_brief = generate_pickup_brief(waste_desc, qty, r_city)
    with st.expander("Smart Pickup Brief", expanded=False):
        st.write(pickup_brief)
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded photo", use_container_width=True)
        st.caption("Optional: use AI to suggest a route, or submit directly for admin review.")
        if st.button("Analyze with AI (optional)", key="donor_analyze"):
            image_bytes = uploaded_file.getvalue()
            st.session_state.tmp_image_b64 = base64.b64encode(image_bytes).decode()
            with st.spinner("Analyzing image (AI)..."):
                st.session_state.ai_result = classify_image_with_ai(image_bytes, waste_desc)
            if ai_verification_passed(st.session_state.ai_result):
                if st.session_state.ai_result.get("source") == "offline_fallback":
                    st.warning(
                        "Hugging Face could not be reached, so MealMatch used the "
                        "offline safety review. Admin approval is still required."
                    )
                else:
                    st.success("Hugging Face AI classified the image. Please review the suggestions below.")
            else:
                st.error(
                    "Hugging Face AI did not classify this image. "
                    f"Reason: {st.session_state.ai_result.get('notes')}"
                )
    else:
        st.error("Photo upload is required.")

    ai = st.session_state.get("ai_result")
    st.subheader("Routing Review")
    if ai:
        st.caption(f"Review source: {ai.get('source', 'huggingface_api')}")
    else:
        st.caption("No AI suggestion yet. Admin can still review and approve this report.")
    suggested_human = ai.get("edible_human") if ai else True
    suggested_compost = ai.get("compost") if ai else False
    suggested_animal = ai.get("edible_animal") if ai else False

    e_human = st.checkbox("Safe for humans", value=suggested_human, key="donor_human")
    e_animal = st.checkbox("Safe for animals", value=suggested_animal, key="donor_animal")
    e_compost = st.checkbox("Safe for composting", value=suggested_compost, key="donor_compost")

    safety_notes_default = ai.get("notes") if ai else pickup_brief
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
                "status": "Pending Admin Approval",
                "claimed_by": None,
                "claimed_by_phone": None,
                "image_b64": st.session_state.get("tmp_image_b64"),
                "ai_review": json.dumps(ai, indent=2) if ai else None,
                "admin_approved": False,
                "ai_verified": ai_verification_passed(ai),
            }
            st.session_state.reports.append(new_report)
            add_chat_message(
                new_id,
                r_name,
                "Restaurant / Donor",
                "New report submitted for admin approval.",
            )
            st.success("Report added successfully! Admin can review it in the dashboard.")
            st.balloons()
            st.session_state.tmp_image_b64 = None
            st.session_state.ai_result = None

    st.subheader("My Submitted Reports")
    my_reports = [
        report
        for report in st.session_state.reports
        if report.get("restaurant") == r_name
    ]
    if not my_reports:
        st.write("No submissions from this restaurant yet.")
    else:
        for report in my_reports:
            st.write(
                f"#{report['id']} - {report['waste_description']} - "
                f"{report['status']}"
            )
            if report.get("admin_approved"):
                st.success("Accepted by admin and visible to volunteers.")
            if report.get("claimed_by"):
                st.info(f"Claimed by volunteer/shelter: {report['claimed_by']}")
                if st.button(
                    f"Contact volunteer for report #{report['id']}",
                    key=f"donor_contact_{report['id']}",
                ):
                    st.session_state.active_message_form = f"donor_{report['id']}"
                    st.rerun()
            if st.session_state.get("active_message_form") == f"donor_{report['id']}":
                render_report_chat(
                    report["id"],
                    (
                        f"Hi {report.get('claimed_by')}, this is {r_name}. "
                        "Let's coordinate this pickup."
                    ),
                    "donor",
                )


def main():
    st.set_page_config(
        page_title="MealMatch - Role-based Access",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_app_theme()

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
