import streamlit as st
import pandas as pd
import pickle
import re
import difflib
import os
from preprocess import clean_ingredients

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="NutriNLP - Intelligent Nutrition", layout="wide")

# --- 2. LOAD DATA & MODELS ---
@st.cache_resource
def load_tools():
    model_path = 'models/best_model.pkl'
    tfidf_path = 'models/tfidf.pkl'
    le_path = 'models/encoder.pkl'
    csv_path = 'data/processed/cleaned_ingredients.csv'
    
    if not os.path.exists(model_path):
        st.error("Model files not found! Please run your training scripts first.")
        return None, None, None, None

    model = pickle.load(open(model_path, 'rb'))
    tfidf = pickle.load(open(tfidf_path, 'rb'))
    le = pickle.load(open(le_path, 'rb'))
    metadata = pd.read_csv(csv_path)
    
    # DYNAMICALLY GET THE MODEL NAME
    # This turns 'KNeighborsClassifier' into 'K-Neighbors Classifier'
    raw_name = type(model).__name__
    model_display_name = re.sub(r"(\w)([A-Z])", r"\1 \2", raw_name)
    
    return model, tfidf, le, metadata, model_display_name

model, tfidf, le, metadata, model_name = load_tools()

# --- 3. EXTENSIVE MANUAL SWAP DATABASE ---
HEALTHY_SWAPS = {
    "butter": ["Olive Oil", "Greek Yogurt", "Mashed Avocado"],
    "mayonnaise": ["Mustard", "Plain Greek Yogurt", "Mashed Avocado"],
    "white rice": ["Brown Rice", "Wild Rice", "Quinoa", "Cauliflower Rice"],
    "rice": ["Brown Rice", "Quinoa"],
    "pasta": ["Whole Wheat Pasta", "Zucchini Noodles (Zoodles)", "Spaghetti Squash"],
    "spaghetti": ["Whole Wheat Spaghetti", "Zucchini Noodles"],
    "noodles": ["Soba Noodles", "Rice Noodles"],
    "white bread": ["Whole Wheat Bread", "Rye Bread", "Multigrain Bread"],
    "bread": ["Whole Wheat Bread", "Lettuce Wraps"],
    "bagel": ["Whole Wheat English Muffin"],
    "tortilla": ["Corn Tortilla", "Lettuce Wraps"],
    "potato chips": ["Air-popped Popcorn", "Kale Chips", "Apple Slices"],
    "chips": ["Baked Vegetable Chips", "Roasted Chickpeas"],
    "french fries": ["Baked Sweet Potato Wedges", "Roasted Carrots"],
    "bacon": ["Turkey Bacon", "Smoked Salmon", "Canadian Bacon"],
    "sausage": ["Chicken Sausage", "Turkey Sausage", "Lean Ground Turkey"],
    "pepperoni": ["Turkey Pepperoni", "Sliced Bell Peppers"],
    "ground beef": ["Lean Ground Turkey", "Ground Chicken", "Lentils"],
    "beef": ["Chicken Breast", "Fish", "Tofu"],
    "pork": ["Chicken Breast", "Fish"],
    "fried chicken": ["Grilled Chicken Breast", "Baked Chicken"],
    "ham": ["Sliced Turkey Breast", "Smoked Tofu"],
    "steak": ["Lean Flank Steak", "Grilled Salmon"],
    "whole milk": ["Skim Milk", "Almond Milk", "Soy Milk"],
    "cheese": ["Feta Cheese", "Low-fat Cottage Cheese", "Nutritional Yeast"],
    "cheddar": ["Low-fat Mozzarella", "Feta"],
    "ice cream": ["Frozen Yogurt", "Fruit Sorbet", "Banana 'Nice' Cream"],
    "sugar": ["Stevia", "Honey", "Maple Syrup"],
    "syrup": ["Fresh Fruit Puree", "Agave Nectar"],
    "soda": ["Sparkling Water with Lemon", "Unsweetened Iced Tea"],
    "juice": ["Fresh Fruit Pieces", "Infused Water"],
    "chocolate": ["Dark Chocolate (70%+)", "Cocoa Nibs"]
}

# --- 4. SIDEBAR UI ---
st.sidebar.title("🥗 Team NutriNLP")
st.sidebar.info(f"""
**Member 1:** Payel Saha (232-35-275)  
**Member 2:** Mst Sayida Parvin (232-35-280)  
**Member 3:** Afifa Rahman (232-35-544)
""")
st.sidebar.markdown("---")
st.sidebar.write("System Status: **🟢 Active**")
# DYNAMICALLY DISPLAY LOADED MODEL NAME
st.sidebar.write(f"Active Engine: **{model_name}**")

# --- 5. MAIN USER INTERFACE ---
st.title("🥗 NutriNLP: Intelligent Food Prediction")
st.markdown("Search for your meal (e.g., **100g white rice, 50g butter**)")

user_input = st.text_input("Meal Description:", placeholder="e.g. 200g pasta and 50g butter")

if user_input:
    matches = re.findall(r'(\d+)\s*g?\s+([a-zA-Z\s]+)', user_input.lower())
    
    total_cal, total_prot, total_fat, total_carb = 0, 0, 0, 0
    breakdown_data, recs = [], []
    all_ingr_names = metadata['ingr_name'].tolist()

    for qty, raw_name in matches:
        qty = float(qty)
        cleaned_search = clean_ingredients(raw_name) 
        closest_match = difflib.get_close_matches(cleaned_search, all_ingr_names, n=1, cutoff=0.3)
        
        if closest_match:
            actual_name = closest_match[0]
            row = metadata[metadata['ingr_name'] == actual_name].iloc[0]
            
            cal = row['cal/g'] * qty
            total_cal += cal
            total_prot += row['protein(g)'] * qty
            total_fat += row['fat(g)'] * qty
            total_carb += row['carb(g)'] * qty
            
            breakdown_data.append({
                "Ingredient": row['ingr_name'], 
                "Quantity": f"{qty}g", 
                "Calories": f"{round(cal, 1)} cal"
            })

            for key in HEALTHY_SWAPS:
                if key in row['ingr_name'].lower():
                    alternatives = ", ".join(HEALTHY_SWAPS[key])
                    recs.append(f"Instead of **{row['ingr_name']}**, consider: **{alternatives}**.")
                    break
        else:
            st.error(f"❌ '{raw_name}' not found in database.")

    # --- 6. DISPLAY RESULTS ---
    if total_cal > 0:
        if total_cal <= 87.49: category, color = "Low", "green"
        elif total_cal <= 260.87: category, color = "Medium", "orange"
        else: category, color = "High", "red"

        st.markdown("---")
        st.markdown(f"<h3 style='text-align: center;'>Meal Category: <span style='color:{color}'>{category}</span></h3>", unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔥 Total Calories", f"{round(total_cal, 1)} cal")
        m2.metric("🍗 Protein", f"{round(total_prot, 1)}g")
        m3.metric("🥑 Fat", f"{round(total_fat, 1)}g")
        m4.metric("🍞 Carbs", f"{round(total_carb, 1)}g")

        st.markdown("---")
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.subheader("📊 Calorie Breakdown")
            st.table(pd.DataFrame(breakdown_data))
        
        with col_right:
            st.subheader("💡 Healthier Recommendations")
            if category == "Low":
                st.success("Your meal choice is already very healthy!")
            else:
                if recs:
                    for r in list(set(recs)):
                        st.warning(r)
                else:
                    st.info("No specific swaps found for your ingredients.")

    # --- 7. ML CLASSIFIER DISPLAY ---
    st.markdown("---")
    with st.expander(f"🔬 View Machine Learning ({model_name}) Analysis"):
        if user_input:
            cleaned_full = clean_ingredients(user_input)
            vec = tfidf.transform([cleaned_full])
            ml_pred = le.inverse_transform(model.predict(vec))[0]
            # USES DYNAMIC MODEL NAME HERE
            st.write(f"The **{model_name}** classifies this linguistic pattern as: **{ml_pred}**")