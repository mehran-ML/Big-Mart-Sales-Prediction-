# import streamlit as st
# import pickle
# import numpy as np

# # Load the trained model
# with open("bigmart_model.pkl", "rb") as file:
#     model = pickle.load(file)

# st.set_page_config(page_title="Big Mart Sales Predictor", layout="centered")
# st.title("🛒 Big Mart Sales Prediction App")
# st.markdown("Enter product and outlet details below to predict expected sales.")

# # Input fields (matching the 11 features in model)
# Item_Identifier = st.number_input("Item Identifier (Encoded)", min_value=0)
# Item_Weight = st.number_input("Item Weight (e.g., 9.3)", min_value=0.0, max_value=50.0, step=0.1)
# Item_Fat_Content = st.selectbox("Item Fat Content (0 = Low Fat, 1 = Regular)", [0, 1])
# Item_Visibility = st.number_input("Item Visibility (e.g., 0.02)", min_value=0.0, max_value=1.0, step=0.01)
# Item_Type = st.number_input("Item Type (Encoded)", min_value=0)
# Item_MRP = st.number_input("Item MRP (e.g., 249.8)", min_value=0.0, max_value=500.0)
# Outlet_Identifier = st.number_input("Outlet Identifier (Encoded)", min_value=0)
# Outlet_Size = st.number_input("Outlet Size (Encoded)", min_value=0)
# Outlet_Location_Type = st.number_input("Outlet Location Type (Encoded)", min_value=0)
# Outlet_Type = st.number_input("Outlet Type (Encoded)", min_value=0)
# Outlet_Establishment_Year = st.number_input("Outlet Establishment Year", min_value=1985, max_value=2025)

# # Predict button
# if st.button("Predict Sales"):
#     input_data = np.array([[
#         Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility,
#         Item_Type, Item_MRP, Outlet_Identifier, Outlet_Size,
#         Outlet_Location_Type, Outlet_Type, Outlet_Establishment_Year
#     ]])

#     prediction = model.predict(input_data)[0]
#     st.success(f"Predicted Sales Amount: ₹ {round(prediction, 2)}")





import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for attractive styling
def load_custom_css():
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .error-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(255, 107, 107, 0.3);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(254, 202, 87, 0.3);
    }
    
    .info-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .sidebar .stSelectbox {
        background: white;
        border-radius: 10px;
    }
    
    h1 {
        color: #2d3436;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #636e72;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

class ModelManager:
    """Handles model loading and prediction with comprehensive error handling"""
    
    def __init__(self, model_path="bigmart_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the trained model with error handling"""
        try:
            with open(self.model_path, "rb") as file:
                self.model = pickle.load(file)
            self.is_loaded = True
            logger.info("Model loaded successfully")
            return True, "Model loaded successfully"
        except FileNotFoundError:
            error_msg = f"Model file '{self.model_path}' not found. Please ensure the model file exists."
            logger.error(error_msg)
            return False, error_msg
        except pickle.UnpicklingError:
            error_msg = "Error loading model: The model file appears to be corrupted."
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error loading model: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def validate_input(self, input_data):
        """Validate input data with comprehensive checks"""
        try:
            # Check if input is numpy array
            if not isinstance(input_data, np.ndarray):
                return False, "Input must be a numpy array"
            
            # Check shape
            if input_data.shape != (1, 11):
                return False, f"Input shape must be (1, 11), got {input_data.shape}"
            
            # Check for NaN or infinite values
            if np.isnan(input_data).any():
                return False, "Input contains NaN values"
            
            if np.isinf(input_data).any():
                return False, "Input contains infinite values"
            
            # Validate specific field ranges
            item_weight = input_data[0][1]
            item_visibility = input_data[0][3]
            item_mrp = input_data[0][5]
            outlet_year = input_data[0][10]
            
            if item_weight < 0 or item_weight > 50:
                return False, "Item weight must be between 0 and 50"
            
            if item_visibility < 0 or item_visibility > 1:
                return False, "Item visibility must be between 0 and 1"
            
            if item_mrp < 0 or item_mrp > 500:
                return False, "Item MRP must be between 0 and 500"
            
            current_year = datetime.now().year
            if outlet_year < 1985 or outlet_year > current_year:
                return False, f"Outlet establishment year must be between 1985 and {current_year}"
            
            return True, "Input validation successful"
            
        except Exception as e:
            return False, f"Input validation error: {str(e)}"
    
    def predict(self, input_data):
        """Make prediction with comprehensive error handling"""
        try:
            if not self.is_loaded:
                return None, "Model not loaded"
            
            # Validate input
            is_valid, validation_msg = self.validate_input(input_data)
            if not is_valid:
                return None, validation_msg
            
            # Make prediction
            prediction = self.model.predict(input_data)
            
            # Validate prediction output
            if prediction is None or len(prediction) == 0:
                return None, "Model returned empty prediction"
            
            predicted_value = float(prediction[0])
            
            # Check if prediction is reasonable
            if predicted_value < 0:
                logger.warning(f"Model predicted negative sales: {predicted_value}")
                return predicted_value, "Warning: Predicted negative sales (unusual but possible)"
            
            if predicted_value > 100000:
                logger.warning(f"Model predicted very high sales: {predicted_value}")
                return predicted_value, "Warning: Predicted unusually high sales"
            
            logger.info(f"Successful prediction: {predicted_value}")
            return predicted_value, "Prediction successful"
            
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
            return None, error_msg

def display_error(message):
    """Display error message with custom styling"""
    st.markdown(f'<div class="error-card">❌ <strong>Error:</strong> {message}</div>', 
                unsafe_allow_html=True)

def display_warning(message):
    """Display warning message with custom styling"""
    st.markdown(f'<div class="warning-card">⚠️ <strong>Warning:</strong> {message}</div>', 
                unsafe_allow_html=True)

def display_info(message):
    """Display info message with custom styling"""
    st.markdown(f'<div class="info-card">ℹ️ <strong>Info:</strong> {message}</div>', 
                unsafe_allow_html=True)

def display_success(prediction):
    """Display prediction result with custom styling"""
    st.markdown(f"""
    <div class="prediction-card">
        <h2>🎯 Prediction Result</h2>
        <h1>₹ {prediction:,.2f}</h1>
        <p>Predicted Sales Amount</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Big Mart Sales Predictor",
        page_icon="🛒",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Header
    st.markdown('<h1>🛒 Big Mart Sales Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered sales prediction for retail optimization</p>', 
                unsafe_allow_html=True)
    
    # Initialize model manager
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    model_manager = st.session_state.model_manager
    
    # Load model if not already loaded
    if not model_manager.is_loaded:
        with st.spinner("Loading prediction model..."):
            success, message = model_manager.load_model()
            if not success:
                display_error(message)
                st.stop()
            else:
                display_info("Model loaded successfully! Ready for predictions.")
    
    # Sidebar for advanced options
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model information
        st.subheader("Model Info")
        if model_manager.is_loaded:
            st.success("✅ Model Loaded")
        else:
            st.error("❌ Model Not Loaded")
        
        # Input validation toggle
        strict_validation = st.checkbox("Strict Input Validation", value=True)
        
        # Prediction confidence
        show_warnings = st.checkbox("Show Prediction Warnings", value=True)
    
    # Main input form
    st.subheader("📝 Product & Outlet Details")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Product Information**")
            item_identifier = st.number_input(
                "Item Identifier (Encoded)", 
                min_value=0, 
                max_value=1000,
                help="Unique identifier for the product (0-1000)"
            )
            
            item_weight = st.number_input(
                "Item Weight (kg)", 
                min_value=0.0, 
                max_value=50.0, 
                value=12.0,
                step=0.1,
                help="Weight of the product in kilograms"
            )
            
            item_fat_content = st.selectbox(
                "Item Fat Content", 
                options=[0, 1],
                format_func=lambda x: "Low Fat" if x == 0 else "Regular",
                help="Fat content level of the product"
            )
            
            item_visibility = st.number_input(
                "Item Visibility", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.05,
                step=0.001,
                format="%.3f",
                help="Visibility percentage of product in store (0-1)"
            )
            
            item_type = st.number_input(
                "Item Type (Encoded)", 
                min_value=0,
                max_value=20,
                help="Product category encoded as number"
            )
            
            item_mrp = st.number_input(
                "Item MRP (₹)", 
                min_value=0.0, 
                max_value=500.0,
                value=150.0,
                step=0.1,
                help="Maximum Retail Price of the product"
            )
        
        with col2:
            st.markdown("**Outlet Information**")
            outlet_identifier = st.number_input(
                "Outlet Identifier (Encoded)", 
                min_value=0,
                max_value=20,
                help="Unique identifier for the outlet"
            )
            
            outlet_size = st.selectbox(
                "Outlet Size", 
                options=[0, 1, 2],
                format_func=lambda x: ["Small", "Medium", "High"][x],
                help="Size category of the outlet"
            )
            
            outlet_location_type = st.selectbox(
                "Outlet Location Type", 
                options=[0, 1, 2],
                format_func=lambda x: ["Tier 1", "Tier 2", "Tier 3"][x],
                help="Location tier of the outlet"
            )
            
            outlet_type = st.selectbox(
                "Outlet Type", 
                options=[0, 1, 2, 3],
                format_func=lambda x: ["Grocery Store", "Supermarket Type1", 
                                     "Supermarket Type2", "Supermarket Type3"][x],
                help="Type of outlet"
            )
            
            outlet_establishment_year = st.number_input(
                "Outlet Establishment Year", 
                min_value=1985, 
                max_value=datetime.now().year,
                value=2000,
                help="Year when the outlet was established"
            )
        
        # Prediction button
        predict_button = st.form_submit_button("🔮 Predict Sales", use_container_width=True)
    
    # Handle prediction
    if predict_button:
        try:
            # Prepare input data
            input_data = np.array([[
                item_identifier, item_weight, item_fat_content, item_visibility,
                item_type, item_mrp, outlet_identifier, outlet_size,
                outlet_location_type, outlet_type, outlet_establishment_year
            ]])
            
            # Make prediction
            with st.spinner("Analyzing data and generating prediction..."):
                prediction, message = model_manager.predict(input_data)
            
            if prediction is not None:
                display_success(prediction)
                
                # Show warning if enabled and present
                if show_warnings and "Warning" in message:
                    display_warning(message.replace("Warning: ", ""))
                
                # Additional insights
                with st.expander("📊 Prediction Insights"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Prediction Confidence", "High" if prediction > 0 else "Low")
                    
                    with col2:
                        category = "High" if prediction > 3000 else "Medium" if prediction > 1000 else "Low"
                        st.metric("Sales Category", category)
                    
                    with col3:
                        profit_margin = prediction * 0.15  # Assuming 15% margin
                        st.metric("Est. Profit", f"₹{profit_margin:.2f}")
                
            else:
                display_error(message)
                
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
            display_error(error_msg)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #636e72;'>"
        "🤖 Powered by XGBoost ML Algorithm | Built with Streamlit"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()