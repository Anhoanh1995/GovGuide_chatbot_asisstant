import streamlit as st
from pages import chatbot ##,audio_chat  # Import chatbot page

st.set_page_config(page_title="My Chatbot", layout="wide")

st.markdown("""
    <div style='background-color: rgb(37, 11, 167); padding: 1rem; margin-bottom: 2rem; border-radius: 10px;'>
        <h2 style='margin: 0; text-align: center; color: white;'>🤖 GovGuide AI Chatbot</h2>
    </div>
""", unsafe_allow_html=True)

# Inject CSS to hide the default sidebar navigation
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar - Custom Navigation Using Buttons
st.sidebar.header("📖 Table of Contents")

# Initialize session state for navigation
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Giới thiệu"  # Default to Home

# Define buttons for navigation
if st.sidebar.button("🏠 Giới thiệu"):
    st.session_state.selected_page = "Giới thiệu"

if st.sidebar.button("💬 Bắt đầu chat"):
    st.session_state.selected_page = "Bắt đầu chat"

if st.sidebar.button("⚙️ Tùy chỉnh"):
    st.session_state.selected_page = "Tùy chỉnh"

if st.sidebar.button("ℹ️ Thông tin về app"):
    st.session_state.selected_page = "Thông tin về app"

if st.sidebar.button("🔊 Chat âm thanh"):
    st.session_state.selected_page = "Chat âm thanh"

# Load the selected page
if st.session_state.selected_page == "Giới thiệu":
    st.title("🏠 Home Page")
    st.write("Welcome to the home page!")

elif st.session_state.selected_page == "Bắt đầu chat":
    st.title("🔊 Start chat page")
    chatbot.render_chat_UI()  # Call the chatbot function from chatbot.py

elif st.session_state.selected_page == "Tùy chỉnh":
    st.title("⚙️ Settings Page")
    st.write("Adjust your preferences here.")

elif st.session_state.selected_page == "Thông tin về app":
    st.title("ℹ️ About Page")
    st.write("Information about this app.")

# elif st.session_state.selected_page == "Chat âm thanh":
#     st.title("🔊 Chat Sound Page")
#     audio_chat.app_sst()  # Call the audio chat function from audio_chat.py
