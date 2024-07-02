import streamlit as st
import joblib
import pandas as pd


import streamlit as st
import joblib
import os

# Function to load the model and scaler
def load_model_and_scaler(model_path, scaler_path):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

# Paths to the models and scalers (relative paths within your repository)
models_info = {
    "Recent Venus": {
        "model_path": "models/RecentVenusDist_model.pkl",
        "scaler_path": "models/HZscaler.pkl"
    },
    "Runaway Greenhouse": {
        "model_path": "models/RunawayGreenhouseDist_model.pkl",
        "scaler_path": "models/HZscaler.pkl"
    },
    "Maximum Greenhouse": {
        "model_path": "models/MaximumGreenhouseDist_model.pkl",
        "scaler_path": "models/HZscaler.pkl"
    },
    "Early Mars": {
        "model_path": "models/EarlyMarsDist_model.pkl",
        "scaler_path": "models/HZscaler.pkl"
    }
}

# Path to the classification model and scaler (relative paths within your repository)
classification_model_path = "models/best_gradient_boosting_modelnew.pkl"
classifier_scaler_path = "models/scalernew.pkl"

# Load the classification model and scaler
classification_model, classifier_scaler = load_model_and_scaler(classification_model_path, classifier_scaler_path)


# Load all models and the common scaler
models = {}
common_scaler = None

for zone, paths in models_info.items():
    model_path = paths["model_path"]
    scaler_path = paths["scaler_path"]
    
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    
    if model:
        models[zone] = model
    
    if scaler and not common_scaler:
        common_scaler = scaler


# Function to predict habitable zone distance
def predict_distance(model, scaler, input_features):
    try:
        features_df = pd.DataFrame([input_features])
        features_df = features_df.fillna(features_df.mean())

        if scaler:
            features_scaled = scaler.transform(features_df)
            prediction = model.predict(features_scaled)[0]
        else:
            prediction = model.predict(features_df)[0]

        return prediction
    except Exception as e:
        st.error(f"Error predicting distance: {e}")
        return None


# Function to get user inputs
def get_user_inputs(input_fields):
    inputs = {}
    for field, default_value in input_fields.items():
        inputs[field] = st.sidebar.number_input(field, value=default_value)
    return inputs


# Custom CSS for button styling and centering
custom_css = """
<style>
.stButton > button:first-child {
    background-color: #BA55D3;
    color: white;
    width: 50%;
    margin-bottom: 10px;
}
.stButton > button:hover {
    background-color: #800080;
}
.centered-buttons {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
</style>
"""

# Apply custom CSS
st.markdown(custom_css, unsafe_allow_html=True)


# Display Header
def niceheader():
    st.markdown("<h1 style='text-align: center; color: purple;'>Exoplanets and the Habitable Zone</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: purple;'>AI Predictive Model for Habitability and Classification</h3>", unsafe_allow_html=True)
    st.markdown("""
    This AI Predictive Model can predict the habitable zone of a star as well as classify exoplanets based on user inputs. Please choose the type of prediction you would like to perform.
    """)


# Display header
niceheader()

# Display buttons centered within a div
st.markdown('<div class="centered-buttons">', unsafe_allow_html=True)

# Button to select Habitable Zone Calculator
if st.button('Habitable Zone Calculator', key='hz_button_1'):
    st.session_state['prediction_type'] = 'hz_calculator'

# Button to select Planet Classifier
if st.button('Planet Classifier', key='classifier_button_1'):
    st.session_state['prediction_type'] = 'classifier'

# Close centered-buttons div
st.markdown('</div>', unsafe_allow_html=True)

# Ensure the rest of the code runs as expected based on user selection
if 'prediction_type' in st.session_state:
    if st.session_state['prediction_type'] == 'hz_calculator':
        st.subheader("Habitable Zone Calculator")
        st.markdown("""
        The habitable zone, also known as the Goldilocks zone, is the region around a star where conditions are just right for liquid water to exist on the surface of a planet.  The four zones predicted in this calculation are Early Venus, Runaway Greenhouse, Maximum Greenhouse and Early Mars.
        """)

        # Sidebar for input features
        st.sidebar.header("Input Features")
        user_features = get_user_inputs({
            "Distance (parsec)": 0.0, "Temperature (K)": 0.0, "Magnitude (mag)": 0.0,
            "Luminosity (solLum)": 0.0, "Mass (solMass)": 0.0, "Radius (solRad)": 0.0, "Parallax (arcsec)": 0.0
        })

        # Button to predict habitable zone
        if st.sidebar.button("Predict", key="hz_predict"):
            predictions = {}
            for zone, model in models.items():
                prediction = predict_distance(model, common_scaler, user_features)
                if prediction is not None:
                    predictions[zone] = prediction
                    st.write(f"{zone}:")
                    st.write(f"  - Distance: {prediction}")
                else:
                    st.error(f"Failed to predict for {zone}")

    elif st.session_state['prediction_type'] == 'classifier':
        st.subheader("Planet Classifier")
        st.markdown("""
        Planet classification refers to categorizing exoplanets based on various characteristics such as size, composition, orbital properties, and atmospheric conditions.  There are 6 general planet types:  Jovian, Subterran, Miniterran, Terran, Superterran and Neptunian.  If you are not able to provide all the user inputs to the left the prediction model will still make a prediction along with the % of liklihood.
        """)

        # Sidebar for input features
        st.sidebar.header("Input Features")
        user_features = get_user_inputs({
            "Planet Radius (Rad)": 0.0, "Planet Mass (Mass)": 0.0, "Planet Density": 0.0,
            "Surface Temperature min (K)": 0.0, "Surface Temperature max (K)": 0.0,
            "Semi-Major Axis (AU)": 0.0, "Orbital Period (days)": 0.0, "Orbital Inclination  (deg°)": 0.0,
            "Surface Gravity (m/s²)": 0.0, "Escape Velocity (km/s)": 0.0, "Star Distance (parsecs)": 0.0,
            "Star Temperature (K)": 0.0, "Radius Error min (Rad)": 0.0, "Radius Error max (Rad)": 0.0,
            "Potential": 0.0, "Eccentricity": 0.0
        })

        feature_mapping = {
            "Planet Radius": "P_RADIUS", "Radius Error (min)": "P_RADIUS_ERROR_MIN", "Radius Error (max)": "P_RADIUS_ERROR_MAX",
            "Planet Mass": "P_MASS", "Planet Density": "P_DENSITY", "Surface Temperature (min)": "P_TEMP_SURF_MIN",
            "Surface Temperature (max)": "P_TEMP_SURF_MAX", "Semi-Major Axis": "P_SEMI_MAJOR_AXIS", "Orbital Period": "P_PERIOD",
            "Orbital Inclination": "P_INCLINATION", "Surface Gravity": "P_GRAVITY", "Escape Velocity": "P_ESCAPE",
            "Potential": "P_POTENTIAL", "Eccentricity": "P_ECCENTRICITY", "Star Distance": "S_DISTANCE", "Star Temperature": "S_TEMPERATURE"
        }

        # Create a DataFrame with the input features in the correct order
        features_df = pd.DataFrame([{feature_mapping[k]: v for k, v in user_features.items()}])
        features_df = features_df[[
            "P_RADIUS", "P_RADIUS_ERROR_MIN", "P_RADIUS_ERROR_MAX", "P_MASS", "P_DENSITY", "P_TEMP_SURF_MIN",
            "P_TEMP_SURF_MAX", "P_SEMI_MAJOR_AXIS", "P_PERIOD", "P_INCLINATION", "P_GRAVITY", "P_ESCAPE", "P_POTENTIAL",
            "P_ECCENTRICITY", "S_DISTANCE", "S_TEMPERATURE"
        ]]

        # Handle missing values
        features_df.fillna(features_df.mean(), inplace=True)

        # Button to predict planet type
        if st.sidebar.button("Predict", key="classify_predict"):
            model = classification_model

            if classifier_scaler:
                features_df_scaled = classifier_scaler.transform(features_df)
            else:
                st.error("Scaler not loaded successfully.")
                features_df_scaled = features_df

            prediction = model.predict(features_df_scaled)[0]
            prediction_proba = model.predict_proba(features_df_scaled).max() * 100

            ptype_mapping = {
                1: 'Jovian',
                2: 'Miniterran',
                3: 'Neptunian',
                4: 'Subterran',
                5: 'Superterran',
                6: 'Terran'
            }

            planet_descriptions = {
                1: 'Jovian: Gas giants similar to Jupiter.',
                2: 'Miniterran: Smaller terrestrial planets.',
                3: 'Neptunian: Ice giants similar to Neptune.',
                4: 'Subterran: Smaller terrestrial planets.',
                5: 'Superterran: Larger terrestrial planets.',
                6: 'Terran: Earth-like planets.'
            }
            
            planet_media = {
                'Jovian': r'C:\Users\Guest User\OneDrive - Torrens Global Education Services\Desktop\Planet Classification Dataset\exoplanet-predictor\planets\Jovian.mp4',
                'Miniterran': r'C:\Users\Guest User\OneDrive - Torrens Global Education Services\Desktop\Planet Classification Dataset\exoplanet-predictor\planets\Miniterran.mp4',
                'Neptunian': r'C:\Users\Guest User\OneDrive - Torrens Global Education Services\Desktop\Planet Classification Dataset\exoplanet-predictor\planets\Neptunian.mp4',
                'Subterran': r'C:\Users\Guest User\OneDrive - Torrens Global Education Services\Desktop\Planet Classification Dataset\exoplanet-predictor\planets\Subterran.mp4',
                'Superterran': r'C:\Users\Guest User\OneDrive - Torrens Global Education Services\Desktop\Planet Classification Dataset\exoplanet-predictor\planets\SuperTerran.mp4',
                'Terran': r'C:\Users\Guest User\OneDrive - Torrens Global Education Services\Desktop\Planet Classification Dataset\exoplanet-predictor\planets\Terran.mp4'
            }

            planet_type = ptype_mapping.get(prediction, "Unknown")
            planet_description = planet_descriptions.get(prediction, "No description available.")
            planet_media_path = planet_media.get(planet_type)

            st.header(f"Predicted Planet Type: {planet_type}")
            st.write(planet_description)
            st.write(f"Prediction Confidence: {prediction_proba:.2f}%")
            if planet_media_path:
                st.video(planet_media_path)
            else:
                st.error("Media file not found.")
