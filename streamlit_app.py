# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import KNNBasic

# Set page config
st.set_page_config(
    page_title="Smart Room Allocation",
    page_icon=":school:",
    layout="wide"
)

# Load models and data
@st.cache_resource
def load_models():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('models/knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    return tfidf, knn_model

@st.cache_data
def load_data():
    students = pd.read_csv('data/students.csv')
    rooms = pd.read_csv('data/rooms.csv')
    allocations = pd.read_csv('data/allocations.csv')
    return students, rooms, allocations

tfidf, knn_model = load_models()
students, rooms, allocations = load_data()

# Sidebar for user input
st.sidebar.header("Student Preferences")
student_id = st.sidebar.selectbox("Student ID", students['student_id'])

# Get student data
student_data = students[students['student_id'] == student_id].iloc[0]

# Display student profile
st.sidebar.subheader("Your Profile")
st.sidebar.write(f"**Academic Program:** {student_data['academic_program']}")
st.sidebar.write(f"**Study Habits:** {student_data['study_habits']}")
st.sidebar.write(f"**Cleanliness:** {student_data['cleanliness']}")
st.sidebar.write(f"**Social Preference:** {student_data['social_preference']}")
st.sidebar.write(f"**Sleep Schedule:** {student_data['sleep_schedule']}")

# Main content
st.title("Smart Room Allocation System")
st.subheader("Find Your Optimal Hostel Room")

# Content-based filtering
def content_based_recommend(student_id):
    student_idx = students[students['student_id'] == student_id].index[0]
    student_features = students.loc[student_idx, 'combined_features']
    student_vector = tfidf.transform([student_features])
    
    room_features = rooms['combined_features']
    room_matrix = tfidf.transform(room_features)
    
    cosine_sim = linear_kernel(student_vector, room_matrix)
    scores = list(enumerate(cosine_sim[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    return scores

# Collaborative filtering
def collaborative_filtering_recommend(student_id):
    all_rooms = rooms['room_id'].unique()
    predictions = [knn_model.predict(str(student_id), str(room_id)) for room_id in all_rooms]
    scores = [(int(pred.iid), pred.est) for pred in predictions]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    return scores

# Hybrid recommendation
def hybrid_recommend(student_id, content_weight=0.6, collab_weight=0.4):
    content_scores = dict(content_based_recommend(student_id))
    collab_scores = dict(collaborative_filtering_recommend(student_id))
    
    # Normalize scores
    max_content = max(content_scores.values()) if max(content_scores.values()) != 0 else 1
    max_collab = max(collab_scores.values()) if max(collab_scores.values()) != 0 else 1
    
    normalized_content = {k: v/max_content for k, v in content_scores.items()}
    normalized_collab = {k: v/max_collab for k, v in collab_scores.items()}
    
    # Combine scores
    hybrid_scores = {
        room_id: content_weight * normalized_content.get(room_id, 0) + 
                collab_weight * normalized_collab.get(room_id, 0)
        for room_id in set(content_scores.keys()).union(set(collab_scores.keys()))
    }
    
    # Sort by hybrid score
    sorted_scores = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_scores

# Get recommendations
recommendation_type = st.radio(
    "Recommendation Type",
    ["Hybrid (Recommended)", "Content-Based", "Collaborative Filtering"],
    horizontal=True
)

if st.button("Get Room Recommendations"):
    st.subheader("Recommended Rooms")
    
    if recommendation_type == "Hybrid (Recommended)":
        scores = hybrid_recommend(student_id)
    elif recommendation_type == "Content-Based":
        scores = content_based_recommend(student_id)
    else:
        scores = collaborative_filtering_recommend(student_id)
    
    # Display top 3 recommendations
    for i, (room_id, score) in enumerate(scores[:3], 1):
        room_data = rooms[rooms['room_id'] == room_id].iloc[0]
        
        with st.expander(f"Recommendation #{i}: Room {room_id} (Score: {score:.2f})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Room Type:** {room_data['room_type']}")
                st.write(f"**Location:** {room_data['location']}")
                st.write(f"**Capacity:** {room_data['capacity']}")
                
            with col2:
                st.write("**Facilities:**")
                facilities = room_data['facilities'].split(',')
                for facility in facilities:
                    st.write(f"- {facility.strip()}")
            
            if st.button(f"Select Room {room_id}", key=f"select_{room_id}"):
                st.success(f"Room {room_id} has been allocated to you!")
                # In a real system, this would update the database

# Room search functionality
st.subheader("Browse All Rooms")
search_col, filter_col = st.columns(2)

with search_col:
    search_query = st.text_input("Search rooms by features")

with filter_col:
    room_type_filter = st.selectbox("Filter by room type", ["All"] + list(rooms['room_type'].unique()))

filtered_rooms = rooms.copy()
if search_query:
    filtered_rooms = filtered_rooms[
        filtered_rooms['combined_features'].str.contains(search_query, case=False)
    ]
if room_type_filter != "All":
    filtered_rooms = filtered_rooms[filtered_rooms['room_type'] == room_type_filter]

st.dataframe(
    filtered_rooms[['room_id', 'room_type', 'location', 'capacity', 'facilities']],
    hide_index=True,
    use_container_width=True
)

# Admin section (password protected)
st.sidebar.markdown("---")
if st.sidebar.checkbox("Admin Login"):
    admin_pass = st.sidebar.text_input("Password", type="password")
    if admin_pass == "admin123":  # In real system, use proper auth
        st.sidebar.success("Logged in as admin")
        
        st.subheader("Administration Dashboard")
        
        tab1, tab2, tab3 = st.tabs(["Students", "Rooms", "Allocations"])
        
        with tab1:
            st.dataframe(students, use_container_width=True)
        
        with tab2:
            st.dataframe(rooms, use_container_width=True)
        
        with tab3:
            st.dataframe(allocations, use_container_width=True)
            
            # Add new allocation
            with st.form("new_allocation"):
                st.write("Add New Allocation")
                new_student = st.selectbox("Student", students['student_id'])
                new_room = st.selectbox("Room", rooms['room_id'])
                new_rating = st.slider("Rating", 1, 5, 3)
                
                if st.form_submit_button("Add Allocation"):
                    new_row = pd.DataFrame({
                        'student_id': [new_student],
                        'room_id': [new_room],
                        'rating': [new_rating]
                    })
                    allocations = pd.concat([allocations, new_row], ignore_index=True)
                    allocations.to_csv('data/allocations.csv', index=False)
                    st.success("Allocation added!")
                    st.experimental_rerun()