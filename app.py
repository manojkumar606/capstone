import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST

# Load Model
pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))

# Function
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Emotions with more consistent emoji representation
emotions_emoji_dict = {
    "anger": "😠", 
    "disgust": "🤮", 
    "fear": "😨", 
    "happy": "😊", 
    "joy": "😂", 
    "neutral": "😐", 
    "sad": "😔", 
    "sadness": "😔", 
    "shame": "😳", 
    "surprise": "😮"
}

# Enhanced Custom CSS for styling
st.markdown("""
    <style>
        /* Global Styles */
        .stApp {
            background-color: #f8f9fa;
        }
        
        /* Title Styling */
        .main-title {
            color: #2C3E50;
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            padding: 20px 0;
            background: linear-gradient(90deg, #4B6CB7 0%, #182848 100%);
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Subheader Styling */
        .subheader {
            color: #3498DB;
            font-size: 28px;
            font-weight: bold;
            margin-top: 20px;
            padding-left: 10px;
            border-left: 5px solid #3498DB;
        }
        
        /* Button Styling */
        .stButton>button {
            background: linear-gradient(90deg, #4B6CB7 0%, #182848 100%);
            color: white;
            font-weight: bold;
            padding: 12px 20px;
            border-radius: 8px;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }
        
        /* Text Area Styling */
        .stTextArea>textarea {
            background-color: #EFF3F6;
            border-radius: 8px;
            border: 2px solid #BDC3C7;
            padding: 10px;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .stTextArea>textarea:focus {
            border-color: #3498DB;
            box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.3);
        }
        
        /* Card Container */
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        /* Success Header */
        .success-header {
            background-color: #2ECC71;
            color: white;
            padding: 10px 15px;
            border-radius: 8px 8px 0 0;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        /* Emoji Display */
        .emoji-display {
            font-size: 64px;
            text-align: center;
            margin: 20px 0;
        }
        
        /* Prediction Result */
        .prediction-result {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        
        /* Confidence Score */
        .confidence-score {
            background-color: #F1C40F;
            color: #34495E;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 20px;
        }
        
        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background-color: #2C3E50;
        }
        
        /* Dashboard cards */
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* About page styling */
        .about-section {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .feature-card {
            background-color: #EBF5FB;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #3498DB;
        }
    </style>
""", unsafe_allow_html=True)

# Main Application
def main():
    # App title with custom HTML
    st.markdown('<div class="main-title">Emotion Classifier App</div>', unsafe_allow_html=True)
    
    # Configure sidebar
    with st.sidebar:
        st.image("https://static.streamlit.io/examples/dice.jpg", width=80)  # Replace with your own logo if available
        st.markdown("## Navigation")
        menu = ["Home", "Monitor", "About"]
        choice = st.selectbox("", menu)
        
        st.markdown("---")
        
    
    # Initialize database tables
    create_page_visited_table()
    create_emotionclf_table()

    if choice == "Home":
        add_page_visited_details("Home", datetime.now(IST))
        
        st.markdown('<div class="subheader">Emotion Detection in Text</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <p>Enter your text below and our AI model will analyze the emotional content.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Express yourself here...", height=150, 
                                   placeholder="Enter the text you want to analyze for emotions...")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit_text = st.form_submit_button(label='Analyze Emotions')

        if submit_text:
            if raw_text.strip() == "":
                st.warning("Please enter some text to analyze")
            else:
                # Get predictions
                prediction = predict_emotions(raw_text)
                probability = get_prediction_proba(raw_text)
                
                # Save prediction to database
                add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))
                
                # Display results in a visually appealing way
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="success-header">Original Text</div>', unsafe_allow_html=True)
                    st.write(raw_text)
                    
                    st.markdown('<div class="success-header">Analysis Result</div>', unsafe_allow_html=True)
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.markdown(f'<div class="emoji-display">{emoji_icon}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="prediction-result">{prediction.title()}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align: center;"><span class="confidence-score">Confidence: {np.max(probability):.2f}</span></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="success-header">Probability Distribution</div>', unsafe_allow_html=True)
                    
                    # Prepare dataframe for visualization
                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["Emotions", "Probability"]
                    
                    # Sort by probability for better visualization
                    proba_df_clean = proba_df_clean.sort_values(by="Probability", ascending=False)
                    
                    # Add color scale based on probability values
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(
                        x=alt.X('Probability:Q', axis=alt.Axis(format='.0%')),
                        y=alt.Y('Emotions:N', sort='-x'),
                        color=alt.Color('Probability:Q', scale=alt.Scale(scheme='blues'))
                    ).properties(height=300)
                    
                    st.altair_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now(IST))
        st.markdown('<div class="subheader">Monitor App</div>', unsafe_allow_html=True)
        
        # Add summary metrics 
        predictions_data = pd.DataFrame(view_all_prediction_details(), 
                                        columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
        
        if not predictions_data.empty:
            total_predictions = len(predictions_data)
            avg_confidence = predictions_data['Probability'].mean()
            most_common = predictions_data['Prediction'].value_counts().idxmax()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>Total Predictions</h3>
                    <h2>{}</h2>
                </div>
                """.format(total_predictions), unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>Avg Confidence</h3>
                    <h2>{:.2f}</h2>
                </div>
                """.format(avg_confidence), unsafe_allow_html=True)
                
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h3>Most Common</h3>
                    <h2>{}</h2>
                </div>
                """.format(most_common), unsafe_allow_html=True)

        # Page Metrics with enhanced visualization
        with st.expander("Page Metrics", expanded=True):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), 
                                              columns=['Page Name', 'Time of Visit'])
            
            # Add timeline of visits
            if not page_visited_details.empty:
                st.markdown("### Recent Page Visits")
                st.dataframe(page_visited_details.tail(10), use_container_width=True)
                
                # Page visit counts
                st.markdown("### Page Visit Distribution")
                pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    c = alt.Chart(pg_count).mark_bar().encode(
                        x='Page Name',
                        y='Counts',
                        color=alt.Color('Page Name', scale=alt.Scale(scheme='category10'))
                    ).properties(height=300)
                    st.altair_chart(c, use_container_width=True)
                
                with col2:
                    p = px.pie(pg_count, values='Counts', names='Page Name', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Set3)
                    st.plotly_chart(p, use_container_width=True)

        # Emotion Classification Metrics with enhanced visualization
        with st.expander('Emotion Classifier Metrics', expanded=True):
            if not predictions_data.empty:
                # Timeline of predictions
                st.markdown("### Recent Predictions")
                st.dataframe(predictions_data[['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit']].tail(10), 
                           use_container_width=True)
                
                # Emotion distribution
                st.markdown("### Emotion Distribution")
                prediction_count = predictions_data['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
                
                # Add emojis to the prediction count
                prediction_count['Emoji'] = prediction_count['Prediction'].map(lambda x: emotions_emoji_dict.get(x, ''))
                prediction_count['Label'] = prediction_count['Prediction'] + ' ' + prediction_count['Emoji']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    pc = alt.Chart(prediction_count).mark_bar().encode(
                        x=alt.X('Counts:Q'),
                        y=alt.Y('Prediction:N', sort='-x'),
                        color=alt.Color('Prediction:N', scale=alt.Scale(scheme='tableau10'))
                    ).properties(height=300)
                    st.altair_chart(pc, use_container_width=True)
                
                with col2:
                    pie = px.pie(prediction_count, values='Counts', names='Prediction', hole=0.4,
                               color_discrete_sequence=px.colors.qualitative.Bold)
                    st.plotly_chart(pie, use_container_width=True)
                
                # Confidence distribution
                st.markdown("### Confidence Distribution")
                fig = px.histogram(predictions_data, x="Probability", nbins=10,
                                 color_discrete_sequence=['#3498DB'])
                st.plotly_chart(fig, use_container_width=True)

    else:  # About page
        add_page_visited_details("About", datetime.now(IST))
        
        st.markdown('<div class="subheader">About This App</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="about-section">
            <h3>Welcome to the Emotion Detection in Text App!</h3>
            <p>Our advanced AI-powered tool helps you understand the emotional content hidden within text.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="subheader">Our Mission</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="about-section">
            <p>At Emotion Detection in Text, our mission is to provide a user-friendly and efficient tool that helps individuals and organizations understand the emotional content hidden within text. We believe that emotions play a crucial role in communication, and by uncovering these emotions, we can gain valuable insights into the underlying sentiments and attitudes expressed in written text.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="subheader">How It Works</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="about-section">
            <p>When you input text into the app, our system processes it and applies advanced natural language processing algorithms to extract meaningful features from the text. These features are then fed into the trained model, which predicts the emotions associated with the input text. The app displays the detected emotions, along with a confidence score, providing you with valuable insights into the emotional content of your text.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="subheader">Key Features</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>⚡ Real-time Emotion Detection</h4>
                <p>Our app offers real-time emotion detection, allowing you to instantly analyze the emotions expressed in any given text.</p>
            </div>
            
            <div class="feature-card">
                <h4>📊 Confidence Score</h4>
                <p>Alongside the detected emotions, our app provides a confidence score, indicating the model's certainty in its predictions.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>🧠 Advanced AI Model</h4>
                <p>Powered by state-of-the-art natural language processing techniques to provide accurate emotion detection.</p>
            </div>
            
            <div class="feature-card">
                <h4>📱 User-friendly Interface</h4>
                <p>We've designed our app with simplicity and usability in mind. The intuitive user interface allows you to effortlessly input text, view the results, and interpret the emotions detected.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="subheader">Applications</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="about-section">
            <ul>
                <li><strong>Social Media Analysis:</strong> Understand the emotional tone of posts, comments, and messages on social platforms</li>
                <li><strong>Customer Feedback:</strong> Analyze reviews and feedback to gauge customer satisfaction and sentiment</li>
                <li><strong>Content Creation:</strong> Ensure your content evokes the intended emotional response</li>
                <li><strong>Market Research:</strong> Gain insights into consumer emotions regarding products or services</li>
                <li><strong>Mental Health:</strong> Track emotional patterns in personal journals or communications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()