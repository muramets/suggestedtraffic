import streamlit as st
import pandas as pd
import re
import os
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import urllib.parse

# Set page configuration
st.set_page_config(
    page_title="YouTube Traffic Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

download_nltk_resources()

# Set page title and custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .stDataFrame {
        width: 100%;
        overflow-x: auto;
    }
    table {
        width: 100%;
        min-width: 1200px;
    }
    th {
        background-color: #1E88E5;
        color: white;
        text-align: left;
        padding: 8px;
    }
    td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    tr:hover {
        background-color: #f5f5f5;
    }
    </style>
    <h1 class="main-header">YouTube Video Traffic Analysis</h1>
    """, unsafe_allow_html=True)

# Get YouTube API Key from Streamlit secrets or environment variable
def get_api_key():
    # First try to get from Streamlit secrets
    try:
        return st.secrets["API_KEY"]
    except:
        # If not in secrets, try environment variable
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if api_key:
            return api_key
        else:
            # Fallback to hardcoded key (not recommended for production)
            return "YOUR_API_KEY_HERE"  # Replace this with your actual API key for local testing

# Initialize YouTube API client
@st.cache_resource
def get_youtube_client():
    api_key = get_api_key()
    return build('youtube', 'v3', developerKey=api_key)

youtube = get_youtube_client()

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    """Extract the video ID from a YouTube URL."""
    # Regular expression patterns for different YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/watch\?.*v=)([^&\n?#]+)',
        r'(?:youtube\.com\/shorts\/)([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

# Function to extract video ID from CSV format (YT_RELATED.videoID)
def extract_id_from_csv_format(id_string):
    """Extract video ID from the format YT_RELATED.videoID."""
    match = re.search(r'YT_RELATED\.([a-zA-Z0-9_-]+)', id_string)
    if match:
        return match.group(1)
    return None

# Function to get video details from YouTube API with rate limit handling
def get_video_details(video_id):
    """Get video details from YouTube API with retry logic for rate limits."""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Get video details
            video_response = youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            ).execute()
            
            if not video_response['items']:
                return None
            
            video_data = video_response['items'][0]
            snippet = video_data['snippet']
            
            # Get video tags (if available)
            tags = snippet.get('tags', [])
            
            # Create a dictionary with video details
            video_details = {
                'id': video_id,
                'title': snippet['title'],
                'description': snippet['description'],
                'tags': tags,
                'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                'channel_title': snippet['channelTitle'],
                'published_at': snippet['publishedAt']
            }
            
            return video_details
        
        except HttpError as e:
            if e.resp.status in [403, 429]:  # Rate limit or quota exceeded
                retry_count += 1
                if retry_count < max_retries:
                    st.warning(f"YouTube API rate limit reached. Retrying in {2**retry_count} seconds...")
                    time.sleep(2**retry_count)  # Exponential backoff
                else:
                    st.error("YouTube API quota exceeded. Please try again later or use a different API key.")
                    return None
            else:
                st.error(f"YouTube API error: {e}")
                return None
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None

# Function to preprocess text with support for Russian and English
def preprocess_text(text):
    """Tokenize and preprocess text for comparison with support for Russian and English."""
    if not text:
        return []
    
    # Tokenize text
    tokens = word_tokenize(text.lower())
    
    # Get stopwords for both English and Russian
    stop_words = set()
    try:
        stop_words.update(stopwords.words('english'))
        # Try to get Russian stopwords if available
        try:
            stop_words.update(stopwords.words('russian'))
        except:
            # If Russian stopwords are not available, we'll continue without them
            pass
    except:
        # If stopwords are not available at all, we'll continue without them
        pass
    
    # Remove stopwords and keep only alphabetic tokens (works for both English and Russian)
    # For Russian, we check if the token contains Cyrillic characters
    def is_valid_token(token):
        # Check if token is alphabetic (works for English)
        if token.isalpha():
            return token not in stop_words
        
        # Check for Cyrillic characters (for Russian)
        return any(0x0400 <= ord(char) <= 0x04FF for char in token) and token not in stop_words
    
    tokens = [token for token in tokens if is_valid_token(token)]
    
    return tokens

# Function to calculate similarity between two texts
def calculate_text_similarity(text1, text2):
    """Calculate similarity between two texts."""
    tokens1 = preprocess_text(text1)
    tokens2 = preprocess_text(text2)
    
    if not tokens1 or not tokens2:
        return 0, []
    
    # Find common words
    common_words = set(tokens1).intersection(set(tokens2))
    
    # Calculate similarity percentage
    similarity_percentage = len(common_words) / max(len(set(tokens1)), len(set(tokens2))) * 100
    
    return similarity_percentage, list(common_words)

# Function to calculate similarity between two sets of tags
def calculate_tag_similarity(tags1, tags2):
    """Calculate similarity between two sets of tags."""
    if not tags1 or not tags2:
        return 0, []
    
    # Preprocess tags
    processed_tags1 = []
    for tag in tags1:
        processed_tags1.extend(preprocess_text(tag))
    
    processed_tags2 = []
    for tag in tags2:
        processed_tags2.extend(preprocess_text(tag))
    
    # Find common tags
    common_tags = set(processed_tags1).intersection(set(processed_tags2))
    
    # Calculate similarity percentage
    similarity_percentage = len(common_tags) / max(len(set(processed_tags1)), len(set(processed_tags2))) * 100
    
    return similarity_percentage, list(common_tags)

# Function to calculate overall similarity
def calculate_overall_similarity(title_sim, desc_sim, tag_sim):
    """Calculate overall similarity based on title, description, and tag similarities."""
    # Simple average of all similarities
    return (title_sim + desc_sim + tag_sim) / 3

# Main function
def main():
    st.title("YouTube Video Traffic Analysis")
    
    # Step 1: Get YouTube video URL
    st.header("Step 1: Enter YouTube Video URL")
    video_url = st.text_input("Enter the URL of your YouTube video:")
    
    # Step 2: Upload CSV file
    st.header("Step 2: Upload CSV with Suggested Traffic")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if video_url and uploaded_file:
        # Extract video ID from URL
        video_id = extract_video_id(video_url)
        
        if not video_id:
            st.error("Invalid YouTube URL. Please enter a valid URL.")
            return
        
        # Get details of the original video
        st.subheader("Fetching details for your video...")
        original_video = get_video_details(video_id)
        
        if not original_video:
            st.error("Could not fetch video details. Please check the URL and try again.")
            return
        
        # Display original video details
        st.subheader("Your Video Details")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image(original_video['thumbnail'], width=200)
        
        with col2:
            st.write(f"**Title:** {original_video['title']}")
            st.write(f"**Channel:** {original_video['channel_title']}")
            
            with st.expander("Video Description"):
                st.write(original_video['description'])
            
            with st.expander("Video Tags"):
                if original_video['tags']:
                    st.write(", ".join(original_video['tags']))
                else:
                    st.write("No tags found for this video.")
        
        # Process CSV file
        st.subheader("Processing CSV data...")
        
        try:
            df = pd.read_csv(uploaded_file)
            
            # Skip the first two rows (header and total)
            df = df.iloc[1:].reset_index(drop=True)
            
            # Extract video IDs from the first column
            video_ids = []
            for id_string in df.iloc[:, 0]:
                vid = extract_id_from_csv_format(str(id_string))
                if vid:
                    video_ids.append(vid)
                else:
                    video_ids.append(None)
            
            # Add video IDs to dataframe
            df['video_id'] = video_ids
            
            # Extract other relevant columns
            df['video_title'] = df.iloc[:, 2]
            df['impressions'] = df.iloc[:, 3]
            df['ctr'] = df.iloc[:, 4]
            df['views'] = df.iloc[:, 5]
            df['avg_view_duration'] = df.iloc[:, 6]
            df['watch_time'] = df.iloc[:, 7]
            
            # Filter out rows with invalid video IDs
            df = df[df['video_id'].notna()].reset_index(drop=True)
            
            if df.empty:
                st.error("No valid video IDs found in the CSV file.")
                return
            
            # Fetch details for each video in the CSV
            st.subheader("Fetching details for videos in CSV...")
            progress_bar = st.progress(0)
            
            video_details_list = []
            for i, vid in enumerate(df['video_id']):
                details = get_video_details(vid)
                if details:
                    video_details_list.append(details)
                progress_bar.progress((i + 1) / len(df['video_id']))
            
            # Calculate similarities
            st.subheader("Calculating similarities...")
            
            results = []
            for i, details in enumerate(video_details_list):
                # Calculate title similarity
                title_sim, common_title_words = calculate_text_similarity(
                    original_video['title'], details['title']
                )
                
                # Calculate description similarity
                desc_sim, common_desc_words = calculate_text_similarity(
                    original_video['description'], details['description']
                )
                
                # Calculate tag similarity
                tag_sim, common_tags = calculate_tag_similarity(
                    original_video['tags'], details['tags']
                )
                
                # Calculate overall similarity
                overall_sim = calculate_overall_similarity(title_sim, desc_sim, tag_sim)
                
                # Get CSV data for this video
                csv_row = df[df['video_id'] == details['id']].iloc[0]
                
                # Create result dictionary
                result = {
                    'id': details['id'],
                    'title': details['title'],
                    'url': f"https://www.youtube.com/watch?v={details['id']}",
                    'overall_similarity': overall_sim,
                    'title_similarity': title_sim,
                    'common_title_words': common_title_words,
                    'description_similarity': desc_sim,
                    'common_description_words': common_desc_words,
                    'tag_similarity': tag_sim,
                    'common_tags': common_tags,
                    'description': details['description'],
                    'tags': details['tags'],
                    'impressions': csv_row['impressions'],
                    'ctr': csv_row['ctr'],
                    'views': csv_row['views'],
                    'avg_view_duration': csv_row['avg_view_duration'],
                    'watch_time': csv_row['watch_time']
                }
                
                results.append(result)
            
            # Sort results by overall similarity (descending)
            results.sort(key=lambda x: x['overall_similarity'], reverse=True)
            
            # Display results in a horizontally scrollable table
            st.header("Analysis Results")
            st.write("Videos sorted by overall similarity to your video (highest to lowest)")
            
            # Create a DataFrame for the horizontally scrollable table
            table_data = []
            for result in results:
                table_data.append({
                    'Title': f"<a href='{result['url']}' target='_blank'>{result['title']}</a>",
                    'Overall Similarity (%)': f"{result['overall_similarity']:.2f}",
                    'Tag Similarity (%)': f"{result['tag_similarity']:.2f}",
                    'Common Tags': ", ".join(result['common_tags'][:5]) + ("..." if len(result['common_tags']) > 5 else ""),
                    'Title Similarity (%)': f"{result['title_similarity']:.2f}",
                    'Common Title Words': ", ".join(result['common_title_words'][:5]) + ("..." if len(result['common_title_words']) > 5 else ""),
                    'Description Similarity (%)': f"{result['description_similarity']:.2f}",
                    'Common Description Words': ", ".join(result['common_description_words'][:5]) + ("..." if len(result['common_description_words']) > 5 else ""),
                    'Impressions': result['impressions'],
                    'CTR (%)': result['ctr'],
                    'Views': result['views'],
                    'Avg View Duration': result['avg_view_duration'],
                    'Watch Time (hours)': result['watch_time']
                })
            
            table_df = pd.DataFrame(table_data)
            
            # Display horizontally scrollable table
            st.subheader("Horizontally Scrollable Analysis Table")
            st.markdown(
                """
                <style>
                .stDataFrame {
                    width: 100%;
                    overflow-x: auto;
                }
                </style>
                """, 
                unsafe_allow_html=True
            )
            
            # Convert DataFrame to HTML with clickable links
            html_table = table_df.to_html(escape=False, index=False)
            st.markdown(html_table, unsafe_allow_html=True)
            
            # Display individual video cards for better readability
            st.subheader("Detailed Video Analysis")
            
            for result in results:
                with st.container():
                    st.markdown(f"### [{result['title']}]({result['url']})")
                    
                    metrics_cols = st.columns(4)
                    with metrics_cols[0]:
                        st.metric("Overall Similarity", f"{result['overall_similarity']:.2f}%")
                    with metrics_cols[1]:
                        st.metric("Title Similarity", f"{result['title_similarity']:.2f}%")
                    with metrics_cols[2]:
                        st.metric("Description Similarity", f"{result['description_similarity']:.2f}%")
                    with metrics_cols[3]:
                        st.metric("Tag Similarity", f"{result['tag_similarity']:.2f}%")
                    
                    traffic_cols = st.columns(4)
                    with traffic_cols[0]:
                        st.metric("Impressions", result['impressions'])
                    with traffic_cols[1]:
                        st.metric("CTR", f"{result['ctr']}%")
                    with traffic_cols[2]:
                        st.metric("Views", result['views'])
                    with traffic_cols[3]:
                        st.metric("Watch Time", f"{result['watch_time']} hours")
                    
                    # Expandable sections
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.expander("Common Words"):
                            st.write("**Title Words:**", ", ".join(result['common_title_words']))
                            st.write("**Description Words:**", ", ".join(result['common_description_words']))
                            st.write("**Tags:**", ", ".join(result['common_tags']))
                    
                    with col2:
                        with st.expander("Video Details"):
                            st.write("**Average View Duration:**", result['avg_view_duration'])
                            with st.expander("Full Description"):
                                st.write(result['description'])
                            with st.expander("All Tags"):
                                st.write(", ".join(result['tags']) if result['tags'] else "No tags")
                    
                    st.markdown("---")
            
            # Allow downloading results as CSV
            csv = table_df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="youtube_analysis_results.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
            return

if __name__ == "__main__":
    main()
