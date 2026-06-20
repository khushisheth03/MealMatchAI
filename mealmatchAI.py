import streamlit as st
import pandas as pd
from datetime import datetime
import math
import os
import base64
import json
try:
    from google.cloud import vision
    from google.cloud.vision_v1 import types
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

# Google Cloud Vision Configuration
# Set via environment variable: export GOOGLE_APPLICATION_CREDENTIALS="graceful-goods-500005-s8-cfb0f75b4a16.json"
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "graceful-goods-500005-s8")

# WhatsApp/Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+8511370099")
ADMIN_WHATSAPP_NUMBER = os.getenv("ADMIN_WHATSAPP_NUMBER", "whatsapp:+8511370099")

def send_whatsapp_message(to_number, message):
    """Send WhatsApp message using Twilio."""
    if not TWILIO_AVAILABLE:
        print(f"[WhatsApp] Twilio not available. Demo message to {to_number}: {message}")
        return {"status": "demo", "message": "Twilio SDK not installed"}
    
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        print(f"[WhatsApp] Demo message to {to_number}: {message}")
        return {"status": "demo", "message": "WhatsApp credentials not configured"}

    try:
        # Validate phone number format
        if not to_number or not str(to_number).strip():
            return {"status": "error", "message": "Invalid recipient phone number"}
        
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message,
            to=to_number
        )
        print(f"[WhatsApp] Message sent successfully to {to_number} (SID: {msg.sid})")
        return {"status": "sent", "sid": msg.sid}
    except Exception as e:
        error_msg = f"WhatsApp Error: {str(e)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}


def show_message_form(form_id, form_type="inquiry"):
    """Display and handle message form for volunteers and donors."""
    st.session_state.active_message_form = form_id
    st.session_state.message_form_type = form_type
    st.rerun()

# Simple distance approximation (km)
def calculate_distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111.0


def classify_image_with_ai(image_bytes):
    """Classify food image using Google Cloud Vision API."""
    try:
        if not GOOGLE_VISION_AVAILABLE:
            raise RuntimeError("Google Cloud Vision SDK not installed. Install with: pip install google-cloud-vision")
        
        if not GOOGLE_CREDENTIALS_PATH:
            raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Set it with: export GOOGLE_APPLICATION_CREDENTIALS='/path/to/your/service-account-key.json'")
        
        try:
            # Initialize Vision API client (credentials from environment variable)
            vision_client = vision.ImageAnnotatorClient()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Vision API client: {str(e)}. Make sure your credentials file exists at: {GOOGLE_CREDENTIALS_PATH}")
        
        # Create image object
        image = types.Image(content=image_bytes)
        
        # Create annotation request
        batch_request = types.AnnotateImageRequest(
            image=image,
            features=[
                types.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=20),
                types.Feature(type_=vision.Feature.Type.TEXT_DETECTION, max_results=10),
                types.Feature(type_=vision.Feature.Type.SAFE_SEARCH_DETECTION),
            ]
        )
        
        # Call Vision API
        response = vision_client.annotate_image(batch_request)
        
        # Extract labels and safe search results
        labels = [label.description for label in response.label_annotations]
        safe_search = response.safe_search_annotation
        
        # Analyze food safety based on labels and content
        edible_human = analyze_food_safety_human(labels, safe_search)
        edible_animal = analyze_food_safety_animal(labels)
        compost = analyze_compost_safety(labels)
        
        notes = generate_food_notes(labels, safe_search)
        
        return {
            "edible_human": edible_human,
            "edible_animal": edible_animal,
            "compost": compost,
            "notes": notes,
            "labels_detected": labels[:10],  # Top 10 labels
            "confidences": {
                "edible_human": 0.8 if edible_human else 0.3,
                "edible_animal": 0.8 if edible_animal else 0.3,
                "compost": 0.85 if compost else 0.2,
            },
        }
    except Exception as e:
        error_msg = f"AI Classification Error: {str(e)}"
        print(error_msg)
        return {
            "edible_human": False,
            "edible_animal": True,
            "compost": True,
            "notes": f"Google Cloud Vision analysis failed. {error_msg}. Please verify manually.",
            "labels_detected": [],
            "confidences": {"edible_human": 0.0, "edible_animal": 0.5, "compost": 0.5},
        }


def analyze_food_safety_human(labels, safe_search):
    """Determine if food is safe for human consumption."""
    unsafe_keywords = ['mold', 'rot', 'spoiled', 'expired', 'contaminated', 'toxic', 'poison', 'waste']
    safe_keywords = ['fresh', 'bread', 'fruit', 'vegetable', 'salad', 'pastry', 'baked', 'clean']
    
    label_lower = ' '.join([l.lower() for l in labels])
    
    # Check for unsafe indicators
    for keyword in unsafe_keywords:
        if keyword in label_lower:
            return False
    
    # Check safe search
    if safe_search:
        if safe_search.adult == vision.Likelihood.LIKELY or safe_search.adult == vision.Likelihood.VERY_LIKELY:
            return False
    
    # Check for safe indicators
    for keyword in safe_keywords:
        if keyword in label_lower:
            return True
    
    return len(labels) > 0


def analyze_food_safety_animal(labels):
    """Determine if food is safe for animals."""
    unsafe_for_animals = ['chocolate', 'onion', 'garlic', 'avocado', 'grape', 'xylitol', 'alcohol', 'toxic']
    label_lower = ' '.join([l.lower() for l in labels])
    
    for keyword in unsafe_for_animals:
        if keyword in label_lower:
            return False
    
    return len(labels) > 0


def analyze_compost_safety(labels):
    """Determine if food can be composted."""
    non_compostable = ['plastic', 'metal', 'glass', 'styrofoam', 'container']
    label_lower = ' '.join([l.lower() for l in labels])
    
    for keyword in non_compostable:
        if keyword in label_lower:
            return False
    
    return True


def generate_food_notes(labels, safe_search):
    """Generate descriptive notes about the food."""
    if not labels:
        return "Unable to identify food items in the image."
    
    notes = f"Detected: {', '.join(labels[:5])}. "
    
    if safe_search:
        if safe_search.adult == vision.Likelihood.POSSIBLE:
            notes += "Contains potentially unsafe content. "
    
    notes += "Please verify classifications manually if unsure."
    return notes


def initialize_state():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "dashboard"
    
    if 'setup_shown' not in st.session_state:
        st.session_state.setup_shown = False
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
                "compost": False,
                "edible_animal": True,
                "compost": False,
                "notes": "All within best-by date, no spoilage. Excellent for food banks.",
                "status": "Available",
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
                "compost": False,
                "edible_animal": False,
                "compost": True,
                "notes": "Good for composting",
                "status": "Available",
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
                "compost": True,
                "edible_animal": True,
                "compost": False,
                "notes": "Meat may be spoiled for humans. Safe for pets (no onions/garlic) and livestock after inspection.",
                "status": "Available",
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
                "compost": False,
                "edible_animal": False,
                "compost": False,
                "notes": "Human-safe. Contains chocolate & xylitol - NOT safe for animals.",
                "status": "Claimed",
                "claimed_by": "Local Shelter Volunteer",
                "claimed_by_phone": None,
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
    
    if 'active_message_form' not in st.session_state:
        st.session_state.active_message_form = None
    if 'message_form_type' not in st.session_state:
        st.session_state.message_form_type = None
    if 'message_phone' not in st.session_state:
        st.session_state.message_phone = ""
    if 'message_text' not in st.session_state:
        st.session_state.message_text = ""


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
        
        st.sidebar.divider()
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("← Back to Login", key="back_btn", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.user_role = None
                st.session_state.user_name = None
                st.session_state.tmp_image_b64 = None
                st.session_state.ai_result = None
                st.session_state.current_page = "dashboard"
                st.rerun()
        with col2:
            if st.button("📱 Logout", key="logout_btn", use_container_width=True):
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
                    col_claim1, col_claim2 = st.columns([2, 1])
                    with col_claim1:
                        if st.button(f"✅ Claim pickup #{row['id']}", key=f"claim_{row['id']}", use_container_width=True):
                            found = False
                            for report in st.session_state.reports:
                                if report["id"] == row["id"]:
                                    report["status"] = "Claimed"
                                    report["claimed_by"] = st.session_state.user_name
                                    found = True
                                    break
                            if found:
                                st.success("✅ Pickup claimed!")
                            else:
                                st.warning("⚠️ This pickup was already claimed by another volunteer.")
                            st.rerun()
                    
                    with col_claim2:
                        if st.button(f"💬 Message", key=f"msg_{row['id']}", use_container_width=True):
                            st.session_state.active_message_form = f"volunteer_{row['id']}"
                            st.rerun()
                    
                    # WhatsApp message form - only show if this is the active form
                    if st.session_state.get("active_message_form") == f"volunteer_{row['id']}":
                        st.divider()
                        st.subheader("📱 Send WhatsApp Message to Donor")
                        
                        contact_phone = st.text_input(
                            "Your WhatsApp number",
                            placeholder="whatsapp:+1234567890",
                            key=f"vol_phone_{row['id']}"
                        )
                        msg_text = st.text_area(
                            "Your message",
                            value=f"Hi, I'm interested in claiming the pickup from {row['restaurant']}. Can I contact you about pickup details?",
                            height=100,
                            key=f"vol_text_{row['id']}"
                        )
                        
                        col_send, col_cancel = st.columns(2)
                        with col_send:
                            if st.button("✅ Send Message", key=f"send_vol_{row['id']}", use_container_width=True):
                                if not contact_phone:
                                    st.error("❌ Please enter your WhatsApp number")
                                else:
                                    admin_msg = f"📲 *Volunteer Inquiry*\n\n👤 {st.session_state.user_name}\n📱 {contact_phone}\n\n🏪 {row['restaurant']}\n📍 Report #{row['id']}\n\n💬 Message:\n{msg_text}"
                                    result = send_whatsapp_message(ADMIN_WHATSAPP_NUMBER, admin_msg)
                                    if result["status"] == "sent":
                                        st.success("✅ Message sent to admin!")
                                        st.session_state.active_message_form = None
                                        st.rerun()
                                    elif result["status"] == "demo":
                                        st.info(f"📌 Demo Mode: {result.get('message')}")
                                        st.session_state.active_message_form = None
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed: {result.get('message')}")
                        
                        with col_cancel:
                            if st.button("❌ Cancel", key=f"cancel_vol_{row['id']}", use_container_width=True):
                                st.session_state.active_message_form = None
                                st.rerun()
                        st.divider()
    else:
        st.warning("No opportunities match the selected filters.")

    st.subheader("My Claimed Pickups")
    my_claims = [r for r in st.session_state.reports if r.get("claimed_by") == st.session_state.user_name]
    if my_claims:
        for claim in my_claims:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"- #{claim['id']} {claim['restaurant']} — {claim['waste_description']} ({claim['status']})")
            with col2:
                if st.button("💬", key=f"contact_donor_{claim['id']}", help="Contact donor", use_container_width=True):
                    st.session_state.active_message_form = f"claimed_{claim['id']}"
                    st.rerun()
            
            # Contact donor form - only show if this is the active form
            if st.session_state.get("active_message_form") == f"claimed_{claim['id']}":
                st.divider()
                st.subheader("📱 Send Message to Donor")
                
                donor_msg = st.text_area(
                    "Your message",
                    value=f"Hi, I'm here for the pickup of {claim['waste_description']}. When is convenient for pickup?",
                    height=100,
                    key=f"claimed_text_{claim['id']}"
                )
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if st.button("✅ Send to Admin", key=f"send_claimed_{claim['id']}", use_container_width=True):
                        admin_msg = f"📲 *Pickup Ready*\n\n👤 {st.session_state.user_name} is ready for pickup!\n\n🏪 {claim['restaurant']}\n📍 Report #{claim['id']}\n\n💬 Message:\n{donor_msg}"
                        result = send_whatsapp_message(ADMIN_WHATSAPP_NUMBER, admin_msg)
                        if result["status"] == "sent":
                            st.success("✅ Message sent to admin to relay to donor!")
                            st.session_state.active_message_form = None
                            st.rerun()
                        elif result["status"] == "demo":
                            st.info(f"📌 Demo Mode: {result.get('message')}")
                            st.session_state.active_message_form = None
                            st.rerun()
                        else:
                            st.error(f"❌ Failed: {result.get('message')}")
                
                with col_d2:
                    if st.button("❌ Cancel", key=f"cancel_claimed_{claim['id']}", use_container_width=True):
                        st.session_state.active_message_form = None
                        st.rerun()
                st.divider()
    else:
        st.write("You have not claimed any pickups yet.")


def admin_page():
    st.title("👨‍💼 Admin Dashboard")
    st.info("Review reports, approve images, verify AI classifications, and update statuses.")
    
    st.markdown("### 📋 Approval Workflow")
    st.markdown("1. **Donor submits** food report with photo")
    st.markdown("2. **You approve image** (click '✅ Approve Image')")
    st.markdown("3. **You run AI verification** (click 'AI verify image') to get safety classifications")
    st.markdown("4. **Report is ready** for volunteers to claim")

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
                if st.button(f"✅ Approve Image #{report['id']}", key=f"admin_approve_{report['id']}"):
                    report["admin_approved"] = True
                    st.success("✅ Report image approved! Now you can verify with AI below.")
                    st.rerun()

            if report.get("image_b64") and report.get("admin_approved") and not report.get("ai_verified"):
                if st.button(f"AI verify image #{report['id']}", key=f"ai_verify_{report['id']}"):
                    image_bytes = base64.b64decode(report["image_b64"])
                    with st.spinner("Running AI verification..."):
                        ai_result = classify_image_with_ai(image_bytes)
                    report["ai_review"] = json.dumps(ai_result, indent=2)
                    report["ai_verified"] = True
                    msg = f"✅ *AI Verification Complete*\n\nReport #{report['id']} - {report['restaurant']}\n\n🏠 Humans: {'✅' if ai_result.get('edible_human') else '❌'}\n🐄 Animals: {'✅' if ai_result.get('edible_animal') else '❌'}\n♻️ Compost: {'✅' if ai_result.get('compost') else '❌'}"
                    result = send_whatsapp_message(ADMIN_WHATSAPP_NUMBER, msg)
                    if result["status"] == "error":
                        st.warning(f"AI verification completed but WhatsApp notification failed: {result.get('message', 'Unknown error')}")
                    else:
                        st.success("AI verification completed. Admin notified.")
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
            e_animal = st.checkbox("Safe for animals", value=report.get("edible_animal", False), key=f"edit_animal_{report['id']}")
            e_compost = st.checkbox("Safe for composting", value=report.get("compost", False), key=f"edit_compost_{report['id']}")

            if st.button(f"Save classifications #{report['id']}", key=f"save_class_{report['id']}"):
                report["edible_human"] = e_human
                report["edible_animal"] = e_animal
                report["compost"] = e_compost
                st.success("Classifications updated.")
                st.rerun()


def donor_page():
    st.title("🏪 Restaurant / Donor Reporting")
    st.info("Upload surplus pictures and provide quantity and safety details for volunteers. Admin will verify classifications.")

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
    suggested_compost = ai.get("compost") if ai else False
    suggested_animal = ai.get("edible_animal") if ai else False

    e_human = st.checkbox("Safe for humans", value=suggested_human, key="donor_human")
    e_animal = st.checkbox("Safe for animals", value=suggested_animal, key="donor_animal")
    e_compost = st.checkbox("Safe for composting", value=suggested_compost, key="donor_compost")

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
                "edible_animal": bool(e_animal),
                "compost": bool(e_compost),
                "notes": safety_notes,
                "status": "Available",
                "claimed_by": None,
                "image_b64": st.session_state.get("tmp_image_b64"),
                "ai_review": None,
                "admin_approved": False,
                "ai_verified": False,
            }
            st.session_state.reports.append(new_report)
            msg = f"📢 *New Food Report!*\n\n🏪 {r_name}\n📍 {r_address}, {r_city}\n\n🍲 {waste_desc}\n⚖️ Quantity: {qty} kg\n\n👤 Donor: {st.session_state.user_name}\n🖼️ Image: {'Yes' if st.session_state.get('tmp_image_b64') else 'No'}\n\nHumans: {'✅' if e_human else '❌'}\nAnimals: {'✅' if e_animal else '❌'}\nCompost: {'✅' if e_compost else '❌'}"
            result = send_whatsapp_message(ADMIN_WHATSAPP_NUMBER, msg)
            if result["status"] == "error":
                st.warning(f"Report submitted but WhatsApp notification failed: {result.get('message', 'Unknown error')}")
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

    # Show setup instructions if credentials not configured
    if not st.session_state.get("setup_shown"):
        with st.expander("📋 Setup Instructions (Click to expand)", expanded=False):
            st.markdown("""
            ### 🔧 API Configuration Required
            
            #### Google Cloud Vision API (for Food Recognition)
            1. Create a Google Cloud project: https://console.cloud.google.com
            2. Enable Vision API in your project
            3. Create a Service Account and download JSON key
            4. Set environment variable:
               ```bash
               export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
               export GOOGLE_PROJECT_ID="your-project-id"
               ```
            5. Install SDK: `pip install google-cloud-vision`
            
            #### Twilio WhatsApp API (for Messaging)
            1. Create a Twilio account: https://www.twilio.com
            2. Set up WhatsApp business account
            3. Set environment variables:
               ```bash
               export TWILIO_ACCOUNT_SID="your-account-sid"
               export TWILIO_AUTH_TOKEN="your-auth-token"
               export TWILIO_WHATSAPP_NUMBER="whatsapp:+1234567890"
               export ADMIN_WHATSAPP_NUMBER="whatsapp:+1234567890"
               ```
            """)
            st.session_state.setup_shown = True
    
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




