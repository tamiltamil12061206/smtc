import streamlit as st
import joblib

# Load models and vectorizer
price_model = joblib.load("price_model.pkl")
quantity_model = joblib.load("quantity_model.pkl")
uses_model = joblib.load("uses_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
le_uses = joblib.load("le_uses.pkl")

st.title("Fertilizer Predictor")
st.write("Enter a fertilizer name to predict its price, quantity, and uses.")

# Input from user
fertilizer_name = st.text_input("Fertilizer Name:")

if st.button("Predict"):
    if fertilizer_name:
        # Vectorize the input
        name_vectorized = vectorizer.transform([fertilizer_name])
        
        # Predict values
        predicted_price = price_model.predict(name_vectorized)[0]
        predicted_quantity = quantity_model.predict(name_vectorized)[0]
        predicted_uses_encoded = uses_model.predict(name_vectorized)[0]
        predicted_uses = le_uses.inverse_transform([predicted_uses_encoded])[0]
        
        # Display results
        st.subheader("Prediction Results")
        st.write(f"**Uses:** {predicted_uses}")
        st.write(f"**Price:** ${round(predicted_price, 2)}")
        st.write(f"**Quantity:** {round(predicted_quantity, 2)} kg")
    else:
        st.error("Please enter a fertilizer name.")
