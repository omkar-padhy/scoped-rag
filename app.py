"""Streamlit frontend for RAG system with login and role-based access"""
import csv
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st

from chat_manager import (
    load_user_chats,
    create_new_chat,
    get_active_chat,
    set_active_chat,
    add_message_to_chat,
    get_chat_messages,
    list_user_chats,
    delete_chat,
    get_chat_context_summary,
    export_chat,
)

API_URL = "http://localhost:8000"
USERS_FILE = Path(__file__).parent / "users.csv"
REQUEST_TIMEOUT = 30  # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for improved design (Material Design 3)
# ─────────────────────────────────────────────────────────────────────────────
def load_css():
    st.markdown("""
    <style>
    /* Material Design 3 Color Tokens */
    :root {
        --md-sys-color-primary: #1a73e8;
        --md-sys-color-on-primary: #ffffff;
        --md-sys-color-primary-container: #d3e3fd;
        --md-sys-color-secondary: #00897b;
        --md-sys-color-tertiary: #f9ab00;
        --md-sys-color-surface: #fef7ff;
        --md-sys-color-surface-variant: #e7e0ec;
        --md-sys-color-on-surface: #1d1b20;
        --md-sys-color-outline: #79747e;
        --md-sys-color-error: #b3261e;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling - MD3 Primary */
    .main-header {
        background: linear-gradient(135deg, #1a73e8 0%, #4285f4 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 500;
        letter-spacing: 0;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* User badge - MD3 style */
    .user-badge {
        background: linear-gradient(135deg, #00897b 0%, #26a69a 100%);
        padding: 0.8rem 1.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.12);
    }
    
    .user-badge.admin {
        background: linear-gradient(135deg, #d32f2f 0%, #e57373 100%);
    }
    
    .user-badge.l5 {
        background: linear-gradient(135deg, #c62828 0%, #ef5350 100%);
    }
    
    .user-badge.l4 {
        background: linear-gradient(135deg, #e65100 0%, #ff9800 100%);
    }
    
    .user-badge.l3 {
        background: linear-gradient(135deg, #f9a825 0%, #ffca28 100%);
    }
    
    .user-badge.l2 {
        background: linear-gradient(135deg, #1a73e8 0%, #4285f4 100%);
    }
    
    .user-badge.l1 {
        background: linear-gradient(135deg, #00897b 0%, #26a69a 100%);
    }
    
    /* Login card - MD3 Surface */
    .login-card {
        background: #ffffff;
        padding: 2.5rem;
        border-radius: 28px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        max-width: 400px;
        margin: 2rem auto;
    }
    
    /* Sidebar styling - MD3 Surface Container */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1c1b1f 0%, #2d2d30 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e6e1e5;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #e6e1e5 !important;
    }
    
    /* Chat messages - MD3 style */
    .stChatMessage {
        border-radius: 16px;
        margin-bottom: 1rem;
    }
    
    /* Buttons - MD3 Filled Button */
    .stButton > button {
        border-radius: 20px;
        font-weight: 500;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 1px 2px rgba(0,0,0,0.12);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.16);
    }
    
    /* MD3 Card elevation */
    .stMetric {
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# API Helper Functions (with caching)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)  # Cache for 60 seconds
def fetch_files(user_level: int) -> dict:
    """Fetch files from API with caching"""
    try:
        r = requests.get(
            f"{API_URL}/files", 
            params={"user_level": user_level},
            timeout=REQUEST_TIMEOUT
        )
        if r.ok:
            return r.json()
    except requests.exceptions.RequestException:
        pass
    return {"files": [], "files_with_levels": [], "total_count": 0, "accessible_count": 0}


@st.cache_data(ttl=60)
def fetch_stats() -> dict:
    """Fetch stats from API with caching"""
    try:
        r = requests.get(f"{API_URL}/stats", timeout=REQUEST_TIMEOUT)
        if r.ok:
            return r.json()
    except requests.exceptions.RequestException:
        pass
    return {"total": 0, "by_type": {}}


def clear_file_cache():
    """Clear file-related caches after modifications"""
    fetch_files.clear()
    fetch_stats.clear()


# ─────────────────────────────────────────────────────────────────────────────
# User authentication functions
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)  # Cache users for 5 minutes
def load_users() -> list[dict]:
    """Load users from CSV file"""
    users = []
    if USERS_FILE.exists():
        with open(USERS_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['level'] = int(row['level'])
                users.append(row)
    return users


def authenticate(username: str, password: str) -> dict | None:
    """Check credentials and return user info if valid"""
    users = load_users()
    for user in users:
        if user['username'] == username and user['password'] == password:
            return user
    return None


def get_role_badge_class(role: str) -> str:
    """Get CSS class for role badge"""
    return role.lower()


def get_access_level_text(level: int) -> str:
    """Get human-readable access level"""
    levels = {
        1: "Public",
        2: "Internal",
        3: "Confidential",
        4: "Sensitive",
        5: "Top Secret"
    }
    return levels.get(level, "Unknown")


def get_level_color(level: int) -> str:
    """Get color for access level badge (Material Design 3)"""
    colors = {
        1: "#00897b",  # Teal - Public
        2: "#1a73e8",  # Blue - Internal
        3: "#f9a825",  # Amber - Confidential
        4: "#e65100",  # Orange - Sensitive
        5: "#c62828",  # Red - Top Secret
    }
    return colors.get(level, "#5f6368")


def get_level_indicator(level: int) -> str:
    """Get simple indicator for access level"""
    return f"L{level}"


# ─────────────────────────────────────────────────────────────────────────────
# File Browser Dialog
# ─────────────────────────────────────────────────────────────────────────────
@st.dialog("Document Browser", width="large")
def show_file_browser_dialog(user_level: int, is_admin: bool):
    """Show file browser as a popup dialog"""
    st.markdown(f"**Your Access Level:** {get_access_level_text(user_level)}")
    st.caption("You can only view and search documents at or below your access level.")
    
    st.divider()
    
    # Use cached fetch
    data = fetch_files(user_level)
    files_with_levels = data.get("files_with_levels", [])
    total_count = data.get("total_count", 0)
    accessible_count = data.get("accessible_count", 0)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", total_count)
    with col2:
        st.metric("Accessible", accessible_count)
    with col3:
        st.metric("Restricted", total_count - accessible_count)
    
    st.divider()
    
    # Search/filter
    search_query = st.text_input("Search files...", placeholder="Type to filter files")
    
    # Filter files by search query
    display_files = files_with_levels
    if search_query:
        search_lower = search_query.lower()
        display_files = [
            f for f in files_with_levels 
            if search_lower in f["name"].lower() or 
               search_lower in f.get("keywords", "").lower()
        ]
    
    if display_files:
        st.markdown(f"**Showing {len(display_files)} file(s):**")
        
        for f in display_files:
            file_level = f["level"]
            level_color = get_level_color(file_level)
            keywords = f.get("keywords", "")
            
            with st.container():
                col1, col2, col3 = st.columns([4, 1, 1])
                
                with col1:
                    st.markdown(f"**{f['name']}**")
                    if keywords:
                        st.caption(keywords)
                
                with col2:
                    st.markdown(f"""
                    <span style="
                        background: {level_color}; 
                        color: white; 
                        padding: 4px 12px; 
                        border-radius: 12px;
                        font-size: 0.85rem;
                        font-weight: 600;
                    ">L{file_level}</span>
                    """, unsafe_allow_html=True)
                
                with col3:
                    if is_admin:
                        if st.button("Delete", key=f"dialog_del_{f['name']}", help="Delete file"):
                            try:
                                del_r = requests.delete(
                                    f"{API_URL}/files/{quote(f['name'], safe='')}", 
                                    timeout=REQUEST_TIMEOUT
                                )
                                if del_r.ok:
                                    clear_file_cache()
                                    st.rerun()
                                else:
                                    st.error(f"Failed: {del_r.text}")
                            except requests.exceptions.RequestException as e:
                                st.error(f"Error: {e}")
                
                st.divider()
    else:
        st.info("No files match your search or access level.")


# ─────────────────────────────────────────────────────────────────────────────
# Upload dialog
# ─────────────────────────────────────────────────────────────────────────────
@st.dialog("Upload Files", width="large")
def show_upload_dialog():
    """Dialog for uploading files (admin only)"""
    st.markdown("Upload PDF or image files to the document store.")
    
    uploaded_files = st.file_uploader(
        "Select files to upload",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="visible"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        
        if st.button("Upload All", use_container_width=True, type="primary"):
            success_count = 0
            for uploaded_file in uploaded_files:
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    r = requests.post(f"{API_URL}/upload", files=files, timeout=REQUEST_TIMEOUT)
                    if r.ok:
                        st.success(f"✓ {uploaded_file.name}")
                        success_count += 1
                    else:
                        st.error(f"✗ {uploaded_file.name}: {r.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"✗ {uploaded_file.name}: Cannot connect to API")
                except requests.exceptions.Timeout:
                    st.error(f"✗ {uploaded_file.name}: Request timed out")
                except Exception as e:
                    st.error(f"✗ {uploaded_file.name}: {e}")
            
            if success_count > 0:
                clear_file_cache()
                st.success(f"Uploaded {success_count} file(s) successfully!")


# ─────────────────────────────────────────────────────────────────────────────
# Login page
# ─────────────────────────────────────────────────────────────────────────────
def show_login_page():
    """Display the login page with full screen colorful design"""
    
    # Full screen background - dark blue/teal gradient
    st.markdown("""
    <style>
    /* Full screen gradient background - dark modern */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }
    
    /* Hide default header */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Center everything vertically and horizontally */
    .block-container {
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        min-height: 100vh !important;
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    .stForm {
        width: 100%;
    }
    
    /* Animated background shapes - cyan/teal */
    .bg-shapes {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        z-index: -1;
    }
    
    .shape {
        position: absolute;
        border-radius: 50%;
        opacity: 0.12;
        animation: float 20s infinite ease-in-out;
    }
    
    .shape-1 {
        width: 600px;
        height: 600px;
        background: linear-gradient(45deg, #00d9ff, #00ffc8);
        top: -200px;
        right: -200px;
    }
    
    .shape-2 {
        width: 500px;
        height: 500px;
        background: linear-gradient(45deg, #7f00ff, #e100ff);
        bottom: -150px;
        left: -150px;
        animation-delay: -7s;
    }
    
    .shape-3 {
        width: 350px;
        height: 350px;
        background: linear-gradient(45deg, #00ffc8, #00d9ff);
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        animation-delay: -12s;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0) rotate(0deg) scale(1); }
        50% { transform: translateY(-40px) rotate(15deg) scale(1.05); }
    }
    
    /* Big centered header */
    .login-header {
        text-align: center;
        padding: 0 0 2.5rem 0;
    }
    
    .brand-title {
        font-size: 5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d9ff, #00ffc8, #00d9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
    }
    
    .brand-subtitle {
        color: rgba(255, 255, 255, 0.85);
        font-size: 1.6rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* Big input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 3px solid rgba(0, 217, 255, 0.4) !important;
        border-radius: 16px !important;
        padding: 20px 24px !important; 
        font-size: 20px !important;
        color: #ffffff !important;
        height: auto !important;
        min-height: 75px !important;
        line-height: normal !important;
        box-sizing: border-box !important;
        outline: none !important;
    }
    
    /* Hide input instructions (Press Enter to apply) */
    div[data-testid="InputInstructions"] {
        display: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
        font-size: 18px !important;
        line-height: normal !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00ffc8 !important;
        box-shadow: 0 0 30px rgba(0, 255, 200, 0.3) !important;
        background: rgba(255, 255, 255, 0.15) !important;
        outline: none !important;
    }
    
    .stTextInput label {
        color: #00d9ff !important;
        font-weight: 700 !important;
        font-size: 20px !important;
        margin-bottom: 12px !important;
        display: block !important;
    }
    
    .stTextInput {
        margin-bottom: 20px !important;
    }
    
    /* Password toggle icon */
    .stTextInput button {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 3px solid rgba(0, 217, 255, 0.4) !important;
        border-radius: 16px !important;
        color: #ffffff !important;
        min-height: 60px !important;
    }
    
    .stSelectbox > div > div > div {
        padding: 18px 24px !important;
        font-size: 20px !important;
        color: #ffffff !important;
    }
    
    .stSelectbox label {
        color: #00d9ff !important;
        font-weight: 700 !important;
        font-size: 20px !important;
        margin-bottom: 12px !important;
    }
    
    .stSelectbox {
        margin-bottom: 20px !important;
    }
    
    /* Big button styling - cyan gradient */
    .stButton > button {
        background: linear-gradient(135deg, #00d9ff 0%, #00ffc8 100%) !important;
        color: #0f0c29 !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 18px 50px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        min-height: 65px !important;
        box-shadow: 0 8px 25px rgba(0, 217, 255, 0.4) !important;
        transition: all 0.3s ease !important;
        margin-top: 20px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.03) !important;
        box-shadow: 0 12px 35px rgba(0, 255, 200, 0.5) !important;
    }
    
    /* Glassmorphism form */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 32px;
        padding: 40px 50px;
        border: 2px solid rgba(0, 217, 255, 0.25);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Remove default container styling if needed */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border: none !important;
    }
    </style>
    
    <!-- Animated background shapes -->
    <div class="bg-shapes">
        <div class="shape shape-1"></div>
        <div class="shape shape-2"></div>
        <div class="shape shape-3"></div>
    </div>
    
    <!-- Big centered header -->
    <div class="login-header">
        <div class="brand-title">🔐 Scoped RAG</div>
        <div class="brand-subtitle">Sign in to access your documents</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Centered form with horizontal fields
    col1, col2, col3 = st.columns([1, 3, 1])
    
    # Company role mappings with levels
    COMPANY_ROLES = {
        "👑 Executive Director (Level 5)": {"username": "admin", "password": "admin123"},
        "🔴 Chief Security Officer (Level 5)": {"username": "user_l5", "password": "pass5@secure"},
        "🟠 Senior Manager (Level 4)": {"username": "user_l4", "password": "pass4@secure"},
        "🟡 Department Head (Level 3)": {"username": "user_l3", "password": "pass3@secure"},
        "🔵 Team Lead (Level 2)": {"username": "user_l2", "password": "pass2@secure"},
        "🟢 Associate (Level 1)": {"username": "user_l1", "password": "pass1@secure"},
    }
    
    with col2:
        # Unified Glass Card Container
        with st.container(border=True):
            # Role selector on top (outside any form logic, immediate update)
            selected_role = st.selectbox(
                "🏢 Select Company Role",
                options=["-- Select Role --"] + list(COMPANY_ROLES.keys()),
                index=0
            )
            
            # Determine default values based on role
            default_user = ""
            default_pass = ""
            if selected_role in COMPANY_ROLES:
                default_user = COMPANY_ROLES[selected_role]["username"]
                default_pass = COMPANY_ROLES[selected_role]["password"]

            # Horizontal username and password (no st.form to allow immediate fill)
            field_col1, field_col2 = st.columns(2)
            with field_col1:
                username = st.text_input("👤 Username", value=default_user, placeholder="Enter username", key="login_user")
            with field_col2:
                password = st.text_input("🔑 Password", type="password", value=default_pass, placeholder="Enter password", key="login_pass")
            
            # Centered big button
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                # Use regular button instead of form submit
                submitted = st.button("🚀 Sign In", use_container_width=True)
            
            if submitted:
                # If using regular inputs/button, we read from state or variables
                # username and password variables are already bound to the text_inputs
                if username and password:
                    user = authenticate(username, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please enter both username and password")


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────
def show_main_app():
    """Display the main application after login"""
    user = st.session_state.user
    username = user['username']
    is_admin = user['role'].lower() == 'admin'
    user_level = user['level']
    
    # Initialize chat state
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
    
    # Load or create active chat
    user_data = load_user_chats(username)
    if not st.session_state.active_chat_id:
        if user_data.get("active_chat_id"):
            st.session_state.active_chat_id = user_data["active_chat_id"]
        elif user_data.get("chats"):
            # Use most recent chat
            st.session_state.active_chat_id = list(user_data["chats"].keys())[0]
        else:
            # Create first chat
            st.session_state.active_chat_id = create_new_chat(username, "New Chat")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Sidebar
    # ─────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        # User info badge - username above role
        badge_class = get_role_badge_class(user['role'])
        st.markdown(f"""
        <div class="user-badge {badge_class}">
            <div style="font-weight: 600; font-size: 1.2rem; margin-bottom: 0.3rem;">{user['display_name']}</div>
            <div style="opacity: 0.9; font-size: 0.85rem;">
                {user['role'].upper()} • Level {user_level} • {get_access_level_text(user_level)}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Chat History Section
        st.header("Chats")
        
        # New Chat button
        if st.button("+ New Chat", use_container_width=True, type="primary"):
            new_chat_id = create_new_chat(username)
            st.session_state.active_chat_id = new_chat_id
            st.session_state.messages = []
            st.rerun()
        
        # List previous chats
        user_chats = list_user_chats(username)
        if user_chats:
            for chat in user_chats[:10]:  # Show last 10 chats
                chat_id = chat["id"]
                title = chat["title"][:30] + ("..." if len(chat["title"]) > 30 else "")
                msg_count = len(chat.get("messages", []))
                
                is_active = chat_id == st.session_state.active_chat_id
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    btn_type = "primary" if is_active else "secondary"
                    if st.button(
                        f"{'→ ' if is_active else ''}{title}", 
                        key=f"chat_{chat_id}",
                        use_container_width=True,
                        type=btn_type if is_active else "secondary"
                    ):
                        st.session_state.active_chat_id = chat_id
                        set_active_chat(username, chat_id)
                        # Load messages from this chat
                        st.session_state.messages = get_chat_messages(username, chat_id)
                        st.rerun()
                
                with col2:
                    if st.button("✕", key=f"del_{chat_id}", help="Delete chat"):
                        delete_chat(username, chat_id)
                        if chat_id == st.session_state.active_chat_id:
                            remaining = list_user_chats(username)
                            if remaining:
                                st.session_state.active_chat_id = remaining[0]["id"]
                                st.session_state.messages = get_chat_messages(username, remaining[0]["id"])
                            else:
                                new_id = create_new_chat(username)
                                st.session_state.active_chat_id = new_id
                                st.session_state.messages = []
                        st.rerun()
        
        st.divider()
        
        # Settings
        st.header("Settings")
        show_sources = st.checkbox("Show sources", value=True)
        show_context = st.checkbox("Show context (debug)", value=False)
        
        st.divider()
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        # Browse Files button
        with col1:
            if st.button("Browse Files", use_container_width=True, help="View all accessible documents"):
                show_file_browser_dialog(user_level, is_admin)
        
        # Upload button (admin only) - opens dialog
        with col2:
            if is_admin:
                if st.button("Upload Files", use_container_width=True, help="Upload new documents"):
                    show_upload_dialog()
            else:
                st.button("Upload Files", use_container_width=True, disabled=True, help="Admin only")
        
        st.divider()
        
        # Sync button (available to all users)
        if st.button("Sync Index", use_container_width=True):
            with st.spinner("Syncing..."):
                try:
                    r = requests.post(f"{API_URL}/reindex", timeout=120)
                    if r.ok:
                        data = r.json()
                        clear_file_cache()
                        st.success(f"Done! ({data.get('status')})")
                    else:
                        st.error(r.text)
                except requests.exceptions.Timeout:
                    st.error("Sync timed out. Try again.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {e}")
        
        st.divider()
        
        # Logout button
        if st.button("Logout", use_container_width=True, type="secondary"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.active_chat_id = None
            st.rerun()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main content area
    # ─────────────────────────────────────────────────────────────────────────
    
    # Get current chat info
    current_chat = get_active_chat(username)
    chat_title = current_chat["title"] if current_chat else "New Chat"
    
    # Header with chat context
    st.markdown(f"""
    <div class="main-header">
        <h1>Scoped RAG</h1>
        <p>{chat_title} • Access Level: {get_access_level_text(user_level)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show chat context summary
    if current_chat:
        context_summary = get_chat_context_summary(username, st.session_state.active_chat_id)
        if context_summary.get("total_queries", 0) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"Queries: {context_summary.get('total_queries', 0)}")
            with col2:
                st.caption(f"Sources: {context_summary.get('unique_sources', 0)}")
            with col3:
                st.caption(f"Model: {context_summary.get('model_used', 'N/A')}")
    
    st.divider()
    
    # Load messages from storage if not in session
    if "messages" not in st.session_state or not st.session_state.messages:
        if st.session_state.active_chat_id:
            st.session_state.messages = get_chat_messages(username, st.session_state.active_chat_id)
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("model"):
                st.caption(f"Model: {msg['model']}")
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- `{s}`")
            if msg.get("context") and show_context:
                with st.expander("Retrieved Context"):
                    chunks = msg["context"].split("\n\n---\n\n")
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.code(chunk[:500] + ("..." if len(chunk) > 500 else ""), language=None)
    
    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        # Add user message to session and storage
        st.session_state.messages.append({"role": "user", "content": question})
        add_message_to_chat(username, st.session_state.active_chat_id, "user", question)
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    endpoint = "/query-with-sources" if show_sources else "/query"
                    
                    # Prepare chat history for context (last 6 messages = 3 turns)
                    recent_history = []
                    for msg in st.session_state.messages[-7:-1]:  # Exclude current question
                        if msg.get("role") in ("user", "assistant"):
                            recent_history.append({
                                "role": msg["role"],
                                "content": msg.get("content", "")
                            })
                    
                    payload = {
                        "question": question,
                        "user_level": user_level,
                        "chat_history": recent_history
                    }
                    r = requests.post(
                        f"{API_URL}{endpoint}", 
                        json=payload,
                        timeout=60
                    )
                    
                    if r.ok:
                        data = r.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])
                        context = data.get("context", "")
                        model = data.get("model", "unknown")
                        
                        st.markdown(answer)
                        st.caption(f"Model: {model}")
                        
                        if show_sources and sources:
                            with st.expander("Sources"):
                                for s in sources:
                                    st.markdown(f"- `{s}`")
                        
                        if show_context and context:
                            with st.expander("Retrieved Context"):
                                chunks = context.split("\n\n---\n\n")
                                for i, chunk in enumerate(chunks, 1):
                                    st.markdown(f"**Chunk {i}:**")
                                    st.code(chunk[:500] + ("..." if len(chunk) > 500 else ""), language=None)
                        
                        # Save to session and storage
                        msg_data = {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources if show_sources else None,
                            "context": context if show_context else None,
                            "model": model,
                        }
                        st.session_state.messages.append(msg_data)
                        
                        # Persist to storage
                        add_message_to_chat(
                            username, 
                            st.session_state.active_chat_id, 
                            "assistant", 
                            answer,
                            sources=sources,
                            context=context if show_context else None,
                            model=model
                        )
                    else:
                        st.error(f"Error: {r.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Please run: `python server.py`")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The query took too long.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Request error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Scoped RAG",
    page_icon="📄",
    )
    
    load_css()
    
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user" not in st.session_state:
        st.session_state.user = None
    
    # Show login or main app
    if st.session_state.logged_in and st.session_state.user:
        show_main_app()
    else:
        show_login_page()


if __name__ == "__main__":
    main()
