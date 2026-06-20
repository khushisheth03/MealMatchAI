import streamlit as st
import pandas as pd
from datetime import datetime
import math
import io
import os
import base64
import json
from PIL import Image

# Simple distance approximation (km)
def calculate_distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111.0


def classify_image_with_ai(image_bytes):
    """Classify an image with highest accuracy. Returns a dict with
    keys: edible_human, edible_animal, compost, notes, confidences.
    For edible_human - allow all types of edible food, for edible_animal, AVOID- CHOCOLATE, GRAPES, RAISINGS, AND ALL OTHER FOOD INAPPROPRATE FOR DOGS AND CATS, for compost- allow all biodegradable substance that are non toxic.
    """
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        b64 = base64.b64encode(image_bytes).decode()
        prompt = (
            "You are a food-safety assistant. Given an image (base64) of food, decide whether it is likely "
            "safe for humans, safe for common pets (dogs/cats), and safe for other animals/livestock. "
            "Respond ONLY with a single valid JSON object with keys: edible_human (true/false), edible_pet (true/false), "
            "edible_animal (true/false), notes (short string), confidences (object with keys edible_human, edible_pet, edible_animal with numbers 0-1).\n\n"
            f"Image (base64, truncated): {b64[:2000]}"
        )

        # Use Chat Completion if available; fall back to Completion if not.
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            text = resp.choices[0].message.content
        except Exception:
            # Older OpenAI clients
            resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=300)
            text = resp.choices[0].text

        # Try to parse JSON from model output
        try:
            parsed = json.loads(text.strip())
            return parsed
        except Exception:
            # If the model didn't return strict JSON, attempt to extract a JSON substring.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(text[start:end+1])
                    return parsed
                except Exception:
                    pass
        # If parsing failed, raise to trigger fallback
        raise RuntimeError("Failed to parse AI response")
    except Exception as e:
        # Conservative fallback: assume human-safe only for clearly fresh bakery/produce is unknown here — default conservative values
        return {
            "edible_human": False,
            "edible_pet": False,
            "edible_animal": True,
            "notes": "AI classification unavailable or failed; please verify manually.",
            "confidences": {"edible_human": 0.0, "edible_pet": 0.0, "edible_animal": 0.5},
        }

st.set_page_config(
    page_title="MealMatch - Food Waste Rescue Map",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Initialize shared data
if 'reports' not in st.session_state:
    st.session_state.reports = [
        {
            "id": 1,
            "restaurant": "Pizza Palace Downtown",
            "address": "XYZ Street, Downtown",
            "city": "Downtown",
            "lat": 40.7128,
            "lon": -74.0060,
            "waste_description": "Fresh bread, pastries & salads - 8 kg",
            "quantity_kg": 8,
            "reported_at": "2026-06-19 10:30",
            "edible_human": True,
            "compost": False,
            "edible_animal": True,
            "notes": "All within best-by date, no spoilage. Excellent for food banks.",
            "status": "Available",
            "claimed_by": None
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
            "compost": True,
            "edible_animal": False,
            "notes": "Good for composting",
            "status": "Available",
            "claimed_by": "ABC SHELTER VOLUNTEER"
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
            "compost": False,
            "edible_animal": True,
            "notes": "Meat may be spoiled for humans. Safe for pets (no onions/garlic) and livestock after inspection.",
            "status": "Available",
            "claimed_by": None
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
            "notes": "Human-safe. Contains chocolate & xylitol - NOT safe for dogs/cats.",
            "status": "Claimed",
            "claimed_by": "Local Shelter Volunteer"
        },
    ]

# Shared user location (sidebar)
if 'user_lat' not in st.session_state:
    st.session_state.user_lat = 40.7128
    st.session_state.user_lon = -74.0060

st.sidebar.header("Your Location (Demo)")
st.session_state.user_lat = st.sidebar.number_input(
    "Latitude", value=st.session_state.user_lat, format="%.4f", key="lat"
)
st.session_state.user_lon = st.sidebar.number_input(
    "Longitude", value=st.session_state.user_lon, format="%.4f", key="lon"
)
st.sidebar.caption("Change these numbers to simulate moving. Nearest spots & distances will update automatically.")

st.title("MealMatch")
st.subheader("1. Find nearest restaurant food waste 2. Check if safe for humans, animals or use as compost 3. Volunteer to rescue & redistribute")

# ========== THREE USER INTERFACES (TABS) ==========
tab1, tab2, tab3 = st.tabs([
    "Find Restaurants & Food Donors",
    "Are you a donor? Report Surplus",
    "Collector / Volunteer Dashboard"
])

# TAB 1: PUBLIC FINDER
with tab1:
    st.header("FIND REGISTERED RESTAURANTS & FOOD DONORS")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        show_available = st.checkbox("Only available spots", value=True)
    with col_f2:
        only_human = st.checkbox("Only animal-edible (food donation)", value=False)
    
    # Build filtered dataframe
    df = pd.DataFrame(st.session_state.reports)
    if show_available:
        df = df[df["status"] == "Available"]
    if only_human:
        df = df[df["edible_animal"] == True]
    
    if not df.empty:
        df["distance_km"] = df.apply(
            lambda r: calculate_distance(
                st.session_state.user_lat, st.session_state.user_lon, r["lat"], r["lon"]
            ), axis=1
        )
        df = df.sort_values("distance_km")
    
    # Map
    st.subheader("Map of Spots")
    if not df.empty:
        map_df = df[["lat", "lon"]].copy()
        map_df.columns = ["latitude", "longitude"]
        st.map(map_df, zoom=11)
    else:
        st.info("No spots match your current filters.")
    
    # Cards with edibility
    st.subheader("Restautants Sorted by Distance from You")
    if not df.empty:
        for _, row in df.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.markdown(f"**{row['restaurant']}**")
                    st.caption(f"{row['address']} • ~{row['distance_km']:.1f} km away")
                    st.write(f"**Waste:** {row['waste_description']} ({row['quantity_kg']} kg)")
                    st.write(f"Reported: {row['reported_at']} | Status: **{row['status']}**")
                with c2:
                    # Edibility badges
                    st.markdown("**Edible for:**")
                    st.write("Humans: " + ("✅ Safe" if row["edible_human"] else "❌ Not recommended"))
                    st.write("Animals: " + ("✅ Safe" if row["edible_animal"] else "❌ Not recommended"))
                    st.write("Compost: " + ("✅ Safe" if row["compost"] else "❌ Not recommended"))
                
                with st.expander("View full safety notes & details"):
                    st.write(f"**Safety Notes:** {row['notes']}")
                    if row["claimed_by"]:
                        st.info(f"Already claimed by: {row['claimed_by']}")
                    if st.button(f"Request Rescue Help for #{row['id']}", key=f"req_{row['id']}"):
                        st.success("Thank you! Volunteers have been notified (demo).")
    else:
        st.warning("No matching spots. Try changing filters or your location in the sidebar.")

# TAB 2: RESTAURANT REPORTER
with tab2:
    st.header("🏪 Report New Surplus / Waste")
    st.info("Help rescue edible food! Clearly mark what is safe for humans, pets, or animals.")
    
    # Reporter UI: allow image upload, AI-assisted classification, then human verification before submit
    uploaded_file = st.file_uploader("Upload photo of surplus (optional)", type=["png", "jpg", "jpeg"])

    r_name = st.text_input("Restaurant / Store Name", "Your Restaurant Name")
    r_address = st.text_input("Full Address", "123 Example Street, Downtown Metro")
    r_city = st.text_input("City / Area", "Downtown Metro")
    r_lat = st.number_input("Latitude (approx)", value=40.71, format="%.4f")
    r_lon = st.number_input("Longitude (approx)", value=-74.01, format="%.4f")

    waste_desc = st.text_area("Describe the surplus/waste", "E.g. Fresh bread, fruits, cooked rice - total 10 kg")
    qty = st.number_input("Quantity (kg)", min_value=1, value=5)

    # AI analyze step
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded photo", use_column_width=True)
        if st.button("🔍 Analyze Photo with AI"):
            image_bytes = uploaded_file.read()
            st.session_state.tmp_image_b64 = base64.b64encode(image_bytes).decode()
            with st.spinner("Analyzing image (AI)..."):
                ai_result = classify_image_with_ai(image_bytes)
            st.session_state.ai_result = ai_result
            st.success("AI has provided a suggested classification — please verify below.")

    ai = st.session_state.get("ai_result")

    st.subheader("Edibility Classification (AI suggestion — verify & adjust)")
    suggested_human = ai.get("edible_human") if ai else True
    suggested_pet = ai.get("edible_pet") if ai else False
    suggested_animal = ai.get("edible_animal") if ai else False

    e_human = st.checkbox("Safe for humans (food banks, shelters, people)", value=suggested_human)
    e_pet = st.checkbox("Safe for pets (dogs, cats - check for toxic ingredients)", value=suggested_pet)
    e_animal = st.checkbox("Safe for other animals / livestock / wildlife", value=suggested_animal)

    safety_notes_default = ai.get("notes") if ai else "All items fresh and within date. No meat spoilage."
    safety_notes = st.text_area("Safety notes / reasons for classification", safety_notes_default)

    # Final verification checkbox to keep humans in the loop
    verified = st.checkbox("I verify that the above classification is correct to the best of my knowledge", value=False)

    if st.button("📤 Submit Verified Report"):
        if not verified:
            st.warning("Please verify the classification before submitting — check the verification box.")
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
                "notes": safety_notes,
                "status": "Available",
                "claimed_by": None,
                "image_b64": st.session_state.get("tmp_image_b64")
            }
            st.session_state.reports.append(new_report)
            st.success("Report added successfully! It is now visible in the Public and Volunteer tabs.")
            st.balloons()

# TAB 3: VOLUNTEER COLLECTOR
with tab3:
    st.header("🤝 Volunteer / Collector Dashboard")
    st.info("Filter by what you can safely handle, claim available surplus, and coordinate rescue.")
    
    v_human = st.checkbox("Show only human-edible (for people/food banks)", value=True)
    v_avail = st.checkbox("Available only (not yet claimed)", value=True)
    
    v_df = pd.DataFrame(st.session_state.reports)
    if v_avail:
        v_df = v_df[v_df["status"] == "Available"]
    if v_human:
        v_df = v_df[v_df["edible_human"] == True]
    
    if not v_df.empty:
        v_df["distance_km"] = v_df.apply(
            lambda r: calculate_distance(
                st.session_state.user_lat, st.session_state.user_lon, r["lat"], r["lon"]
            ), axis=1
        )
        v_df = v_df.sort_values("distance_km")
    
    st.subheader("Available Rescue Opportunities (sorted by distance)")
    if not v_df.empty:
        for _, row in v_df.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['restaurant']}** — {row['address']}")
                st.caption(f"~{row['distance_km']:.1f} km away | {row['quantity_kg']} kg | {row['reported_at']}")
                
                st.write("Humans: " + ("✅" if row["edible_human"] else "❌") + 
                         "Animals: " + ("✅" if row["edible_animal"] else "❌") + 
                         "Compost: " + ("✅" if row["compost"] else "❌"))
                
                with st.expander("Full details & safety notes"):
                    st.write(row["waste_description"])
                    st.write(f"**Notes:** {row['notes']}")
                
                if st.button(f"Claim this pickup #{row['id']}", key=f"claim_{row['id']}"):
                    for report in st.session_state.reports:
                        if report["id"] == row["id"]:
                            report["status"] = "Claimed"
                            report["claimed_by"] = "Volunteer (Demo User)"
                            break
                    st.success("Claimed! In a real app this would notify the restaurant and add to your route.")
                    st.rerun()
    else:
        st.info("No available opportunities matching your filters.")
    
    # My claims
    st.subheader("My Claimed Pickups (this demo session)")
    my_claims = [r for r in st.session_state.reports if r.get("claimed_by") == "Volunteer (Demo User)"]
    if my_claims:
        for c in my_claims:
            st.write(f"- #{c['id']} {c['restaurant']}: {c['waste_description']} ({c['status']})")
    else:
        st.caption("You have not claimed anything yet in this session.")


