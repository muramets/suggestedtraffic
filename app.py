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

# Disable automatic rerunning
st.cache_data(ttl=None)
def get_session_state():
    """Get session state to prevent auto-refresh."""
    return {}

# Initialize session state
get_session_state()

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
        font-size: 1.8rem;
        color: #1E88E5;
    }
    .stDataFrame {
        width: 100%;
        overflow-x: auto;
    }
    table {
        width: 100%;
        min-width: 1200px;
        font-size: 0.9rem;
    }
    th {
        background-color: #1E88E5;
        color: white;
        text-align: left;
        padding: 6px;
        font-size: 0.9rem;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    td {
        padding: 6px;
        border-bottom: 1px solid #ddd;
        font-size: 0.9rem;
    }
    tr:hover {
        background-color: #e3f2fd;
        color: #000;
    }
    tr:hover td {
        color: #000;
    }
    .streamlit-expanderHeader {
        font-size: 0.9rem;
    }
    p, div, span, li {
        font-size: 0.95rem;
    }
    h1 {
        font-size: 1.8rem;
    }
    h2 {
        font-size: 1.5rem;
    }
    h3 {
        font-size: 1.2rem;
    }
    /* Fixed header styles */
    .fixed-header {
        position: sticky;
        top: 0;
        background-color: white;
        z-index: 999;
        border-bottom: 1px solid #ddd;
    }
    /* Make sure the table container has a fixed height to enable scrolling */
    .table-container {
        max-height: 600px;
        overflow-y: auto;
    }
    </style>
    <h1 class="main-header">YouTube Video Traffic Analysis</h1>
    """, unsafe_allow_html=True)

# Initialize YouTube API client
@st.cache_resource
def get_youtube_client(api_key):
    """Create a YouTube API client with the provided API key."""
    if not api_key or api_key.strip() == "" or api_key == "YOUR_API_KEY_HERE":
        return None
    return build('youtube', 'v3', developerKey=api_key)

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
    """Calculate similarity between two sets of tags.
    
    Compare whole tags (not individual words) as requested by the user.
    Tags are considered matching only if they are exactly the same.
    """
    if not tags1 or not tags2:
        return 0, []
    
    # Convert tags to lowercase for case-insensitive comparison
    tags1_lower = [tag.lower().strip() for tag in tags1]
    tags2_lower = [tag.lower().strip() for tag in tags2]
    
    # Find common tags (exact matches only)
    common_tags = set(tags1_lower).intersection(set(tags2_lower))
    
    # Calculate similarity percentage
    similarity_percentage = len(common_tags) / max(len(tags1_lower), len(tags2_lower)) * 100
    
    return similarity_percentage, list(common_tags)

# Function to calculate overall similarity
def calculate_overall_similarity(title_sim, desc_sim, tag_sim):
    """Calculate overall similarity based on title, description, and tag similarities."""
    # Simple average of all similarities
    return (title_sim + desc_sim + tag_sim) / 3

# Function to get video details from YouTube API with rate limit handling
def get_video_details(youtube, video_id):
    """Get video details from YouTube API with retry logic for rate limits and other errors."""
    if not youtube:
        st.error("YouTube API client not initialized. Please enter a valid API key.")
        return None
        
    max_retries = 5  # Increased from 3 to 5
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
            retry_count += 1
            wait_time = 2**retry_count  # Exponential backoff
            
            # Handle different error codes with specific messages
            if e.resp.status in [403, 429]:  # Rate limit or quota exceeded
                if retry_count < max_retries:
                    st.warning(f"YouTube API rate limit reached. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    st.error("YouTube API quota exceeded. Please try again later or use a different API key.")
                    return None
            elif e.resp.status == 500:  # Internal server error
                if retry_count < max_retries:
                    st.warning(f"YouTube API internal error. Retrying in {wait_time} seconds... ({retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    st.error("YouTube API is experiencing internal issues. This is not a problem with your request. Please try again later.")
                    st.info("If this error persists, you may want to try a different API key or check the YouTube API status.")
                    return None
            elif e.resp.status == 404:  # Not found
                st.error(f"Video with ID {video_id} not found. It may have been deleted or made private.")
                return None
            else:
                error_message = str(e)
                st.error(f"YouTube API error: {error_message}")
                
                # Provide more user-friendly explanations for common errors
                if "backendError" in error_message:
                    st.info("This is a temporary issue with YouTube's servers. Please try again in a few minutes.")
                elif "quotaExceeded" in error_message:
                    st.info("Your YouTube API quota has been exceeded. Please try again tomorrow or use a different API key.")
                
                if retry_count < max_retries:
                    st.warning(f"Retrying in {wait_time} seconds... ({retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    return None
        except Exception as e:
            error_message = str(e)
            st.error(f"An error occurred: {error_message}")
            
            # Try to provide more helpful context for the error
            if "socket" in error_message.lower() or "timeout" in error_message.lower() or "connection" in error_message.lower():
                st.info("This appears to be a network connectivity issue. Please check your internet connection and try again.")
            
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2**retry_count
                st.warning(f"Retrying in {wait_time} seconds... ({retry_count}/{max_retries})")
                time.sleep(wait_time)
            else:
                return None

# Кэшируем функцию для получения информации о видео из API
@st.cache_data(ttl=3600)  # Кэшируем на 1 час
def fetch_video_details_for_list(_youtube, video_ids):
    """Fetch details for multiple videos and cache the results."""
    video_details_list = []
    # Удаляем прогресс-бар из этой функции
    for vid in video_ids:
        details = get_video_details(_youtube, vid)
        if details:
            video_details_list.append(details)
    return video_details_list

# Function to convert duration string to seconds
def convert_duration_to_seconds(duration_str):
    if pd.isna(duration_str) or duration_str == '':
        return 0
    
    duration_str = str(duration_str).strip()
    # Handle MM:SS format
    if ':' in duration_str:
        parts = duration_str.split(':')
        if len(parts) == 2:  # MM:SS
            try:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
            except ValueError:
                return 0
        elif len(parts) == 3:  # HH:MM:SS
            try:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            except ValueError:
                return 0
    # Try to convert to float if it's just a number
    try:
        return float(duration_str)
    except ValueError:
        return 0

# Main function
def main():
    # Create sidebar for API key input
    with st.sidebar:
        st.header("YouTube API Configuration")
        api_key = st.text_input(
            "Enter your YouTube API Key",
            type="password",
            help="Get your API key from the Google Cloud Console"
        )
        
        if api_key:
            st.success("API key provided! You can now analyze videos.")
        else:
            st.warning("Please enter your YouTube API key to use this application.")
            st.markdown("""
            ### How to get a YouTube API Key:
            1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
            2. Create a new project or select an existing one
            3. Enable the YouTube Data API v3
            4. Create an API key
            5. Enter the key in the field above
            """)
        
        # Add a note about API usage
        st.info("Note: This application makes API calls only when you submit data, not automatically.")
    
    # Initialize YouTube client with the provided API key
    youtube = get_youtube_client(api_key)
    
    # Step 1: Get YouTube video URL
    st.header("Step 1: Enter YouTube Video URL")
    video_url = st.text_input("Enter the URL of your YouTube video:")
    
    # Step 2: Upload CSV file
    st.header("Step 2: Upload CSV with Suggested Traffic")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    # Check if all inputs are provided
    if not api_key or api_key.strip() == "" or api_key == "YOUR_API_KEY_HERE":
        st.error("Please enter a valid YouTube API key in the sidebar to continue.")
        st.info("The application requires a valid YouTube API key to function.")
        return
    
    # Test the API key with a simple request
    try:
        with st.spinner("Validating API key..."):
            test_response = youtube.channels().list(
                part="snippet",
                id="UC_x5XG1OV2P6uZZ5FSM9Ttw"  # Google Developers channel
            ).execute()
            st.success("API key is valid!")
    except HttpError as e:
        if "API key not valid" in str(e):
            st.error("The API key you entered is not valid. Please check and try again.")
            return
        # Other errors are ok - might be quota or permission issues that won't affect basic functionality
    
    # Модифицируем основную логику для использования session_state
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = None
    
    if video_url and uploaded_file:
        # Extract video ID from URL
        video_id = extract_video_id(video_url)
        
        if not video_id:
            st.error("Invalid YouTube URL. Please enter a valid URL.")
        else:
            # Get details of the original video only once
            if 'original_video' not in st.session_state:
                with st.spinner("Fetching details for your video..."):
                    st.session_state.original_video = get_video_details(youtube, video_id)
            
            original_video = st.session_state.original_video
            
            if not original_video:
                st.error("Could not fetch video details. Please check the URL and try again.")
            else:
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
                        if (original_video['tags']):
                            st.write(", ".join(original_video['tags']))
                        else:
                            st.write("No tags found for this video.")
                
                # Process CSV file only once
                if st.session_state.processed_results is None:
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
                        
                        # Parse avg_view_duration correctly - convert time format to seconds
                        avg_duration_col = df.iloc[:, 6]
                        df['avg_view_duration_seconds'] = avg_duration_col.apply(convert_duration_to_seconds)
                        df['avg_view_duration'] = avg_duration_col  # Keep original formatted string
                        df['watch_time'] = df.iloc[:, 7]
                        
                        # Filter out rows with invalid video IDs
                        df = df[df['video_id'].notna()].reset_index(drop=True)
                        
                        if df.empty:
                            st.error("No valid video IDs found in the CSV file.")
                        else:
                            # Fetch details for each video in the CSV
                            st.subheader("Fetching details for videos in CSV...")

                            # Создаём прогресс-бар здесь
                            progress_bar = st.progress(0, text="Fetching details for videos in CSV...")

                            video_ids = df['video_id'].tolist()
                            video_details_list = []
                            total = len(video_ids)
                            for i, vid in enumerate(video_ids):
                                details = get_video_details(youtube, vid)
                                if details:
                                    video_details_list.append(details)
                                progress_bar.progress((i + 1) / total, text=f"Fetching video {i+1} of {total}...")
                            progress_bar.empty()
                            
                            # Calculate similarities
                            st.subheader("Calculating similarities...")
                            
                            results = []
                            for details in video_details_list:
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
                                    'avg_view_duration_seconds': csv_row['avg_view_duration_seconds'],
                                    'watch_time': csv_row['watch_time']
                                }
                                
                                results.append(result)
                            
                            # Sort results by overall similarity (descending)
                            results.sort(key=lambda x: x['overall_similarity'], reverse=True)
                            
                            # Save processed results to session state
                            st.session_state.processed_results = results
                    
                    except Exception as e:
                        st.error(f"Error processing CSV file: {e}")
                
                # Now display the results from session state
                results = st.session_state.processed_results
                if results:
                    # Function to highlight matching words
                    def highlight_matching_words(text, common_words):
                        if not text or not common_words:
                            return text
                        
                        for word in common_words:
                            # Case-insensitive replacement with green highlight
                            pattern = re.compile(re.escape(word), re.IGNORECASE)
                            text = pattern.sub(f'<span style="color: green; font-weight: bold;">{word}</span>', text)
                        
                        return text
                    
                    # Add a selectbox to choose a video to view details
                    st.subheader("Video Details")
                    video_titles = [result['title'] for result in results]
                    
                    # Используем ключ для стабильности
                    selected_video_index = st.selectbox(
                        "Select a video to view details:",
                        range(len(video_titles)),
                        format_func=lambda i: video_titles[i],
                        key="video_selector"
                    )
                    
                    # Display details for the selected video
                    if selected_video_index is not None:
                        result = results[selected_video_index]
                        
                        st.subheader(f"Details for: {result['title']}")
                        
                        # Display metrics in columns
                        cols = st.columns(4)
                        cols[0].metric("Overall Similarity", f"{result['overall_similarity']:.2f}%")
                        cols[1].metric("Title Similarity", f"{result['title_similarity']:.2f}%")
                        cols[2].metric("Description Similarity", f"{result['description_similarity']:.2f}%")
                        cols[3].metric("Tag Similarity", f"{result['tag_similarity']:.2f}%")
                        
                        # Highlight matching words in description
                        highlighted_description = highlight_matching_words(
                            result['description'], 
                            result['common_description_words']
                        )
                        
                        # Highlight matching tags
                        highlighted_tags = []
                        for tag in result['tags']:
                            if tag.lower().strip() in [t.lower().strip() for t in result['common_tags']]:
                                highlighted_tags.append(f'<span style="color: green; font-weight: bold;">{tag}</span>')
                            else:
                                highlighted_tags.append(tag)
                        
                        # Create expandable sections for description and tags
                        with st.expander("Description"):
                            st.markdown(highlighted_description, unsafe_allow_html=True)
                        
                        with st.expander("Tags"):
                            st.markdown(', '.join(highlighted_tags), unsafe_allow_html=True)
                    
                    # Create a DataFrame for the horizontally scrollable table
                    table_data = []
                    for result in results:
                        table_data.append({
                            'Title': f"<a href='{result['url']}' target='_blank'>{result['title']}</a>",  # Make title clickable with HTML
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
                    
                    
                    # Display results in a horizontally scrollable table with headers
                    st.header("Analysis Results")
                    st.write("Videos sorted by overall similarity to your video (highest to lowest)")
                    st.subheader("Analysis Table (Click column headers to sort)")
                    
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
                    
                    # Prepare DataFrame for display with proper data types for sorting
                    display_df = table_df.copy()
                    
                    # Convert numeric columns to proper numeric types for sorting - exclude Avg View Duration
                    numeric_columns = [
                        'Overall Similarity (%)', 'Tag Similarity (%)', 'Title Similarity (%)', 
                        'Description Similarity (%)', 'Impressions', 'CTR (%)', 'Views', 
                        'Watch Time (hours)'  # Removed 'Avg View Duration' to preserve time format
                    ]
                    
                    for col in numeric_columns:
                        if col in display_df.columns:
                            # Extract numeric values from string columns (remove % and other non-numeric characters)
                            if display_df[col].dtype == 'object':
                                display_df[col] = display_df[col].str.extract(r'([\d\.]+)').astype(float)
                            else:
                                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                    
                    # Create a container with fixed height for scrolling with fixed header
                    st.markdown('<div class="table-container">', unsafe_allow_html=True)
                    
                    # Create a DataFrame for display with proper data types for sorting
                    # and add a Video Link column with Title as the first column
                    display_df = pd.DataFrame()
                    
                    # Add Title as the first column
                    display_df['Title'] = [result['title'] for result in results]
                    
                    # Add all other columns except the HTML Title
                    for col in table_df.columns:
                        if col != 'Title':
                            display_df[col] = table_df[col]
                    
                    # Add Video Link as the last column
                    display_df['Video Link'] = [result['url'] for result in results]
                    
                    # Convert numeric columns to proper numeric types for sorting - exclude Avg View Duration
                    numeric_columns = [
                        'Overall Similarity (%)', 'Tag Similarity (%)', 'Title Similarity (%)', 
                        'Description Similarity (%)', 'Impressions', 'CTR (%)', 'Views', 
                        'Watch Time (hours)'  # Removed 'Avg View Duration' to preserve time format
                    ]
                    
                    for col in numeric_columns:
                        if col in display_df.columns:
                            # Extract numeric values from string columns (remove % and other non-numeric characters)
                            if display_df[col].dtype == 'object':
                                display_df[col] = display_df[col].str.extract(r'([\d\.]+)').astype(float)
                            else:
                                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

                    # --- Состояния для порядка и чекбоксов ---
                    if 'favorites' not in st.session_state:
                        st.session_state.favorites = {}
                    if 'editor_order' not in st.session_state or len(st.session_state.editor_order) != len(results):
                        st.session_state.editor_order = list(range(len(results)))

                    # --- Формируем display_df в порядке editor_order ---
                    ordered_results = [results[i] for i in st.session_state.editor_order]
                    display_df = pd.DataFrame()
                    display_df['Title'] = [r['title'] for r in ordered_results]
                    for col in [
                        'Overall Similarity (%)', 'Tag Similarity (%)', 'Common Tags',
                        'Title Similarity (%)', 'Common Title Words', 'Description Similarity (%)',
                        'Common Description Words', 'Impressions', 'CTR (%)', 'Views',
                        'Avg View Duration', 'Watch Time (hours)', 'Video Link'
                    ]:
                        if col == 'Video Link':
                            display_df[col] = [r['url'] for r in ordered_results]
                        elif col == 'Overall Similarity (%)':
                            display_df[col] = [f"{r['overall_similarity']:.2f}" for r in ordered_results]
                        elif col == 'Tag Similarity (%)':
                            display_df[col] = [f"{r['tag_similarity']:.2f}" for r in ordered_results]
                        elif col == 'Title Similarity (%)':
                            display_df[col] = [f"{r['title_similarity']:.2f}" for r in ordered_results]
                        elif col == 'Description Similarity (%)':
                            display_df[col] = [f"{r['description_similarity']:.2f}" for r in ordered_results]
                        elif col == 'Impressions':
                            display_df[col] = [r['impressions'] for r in ordered_results]
                        elif col == 'CTR (%)':
                            display_df[col] = [r['ctr'] for r in ordered_results]
                        elif col == 'Views':
                            display_df[col] = [r['views'] for r in ordered_results]
                        elif col == 'Avg View Duration':
                            display_df[col] = [r['avg_view_duration'] for r in ordered_results]
                        elif col == 'Watch Time (hours)':
                            display_df[col] = [r['watch_time'] for r in ordered_results]
                        elif col == 'Common Tags':
                            display_df[col] = [", ".join(r['common_tags'][:5]) + ("..." if len(r['common_tags']) > 5 else "") for r in ordered_results]
                        elif col == 'Common Title Words':
                            display_df[col] = [", ".join(r['common_title_words'][:5]) + ("..." if len(r['common_title_words']) > 5 else "") for r in ordered_results]
                        elif col == 'Common Description Words':
                            display_df[col] = [", ".join(r['common_description_words'][:5]) + ("..." if len(r['common_description_words']) > 5 else "") for r in ordered_results]
                    display_df['Избранное'] = [
                        st.session_state.favorites.get(r['id'], False)
                        for r in ordered_results
                    ]

                    # --- Callback для обновления порядка и чекбоксов ---
                    def on_editor_change():
                        editor_state = st.session_state["main_data_editor"]
                        # Сохраняем порядок строк
                        st.session_state.editor_order = editor_state.get("row_indices", list(range(len(ordered_results))))
                        # Сохраняем чекбоксы
                        data = editor_state.get("data", [])
                        for idx, row in enumerate(data):
                            # Получаем id видео по текущему порядку
                            video_id = ordered_results[idx]['id']
                            st.session_state.favorites[video_id] = row.get('Избранное', False)

                    st.markdown(
                        """
                        <style>
                        [data-testid="stDataFrame"] th:nth-child(15) div {
                            white-space: normal !important;
                            min-width: 110px !important;
                            max-width: 160px !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    edited_df = st.data_editor(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Title": st.column_config.Column("Title", width="large"),
                            "Overall Similarity (%)": st.column_config.NumberColumn("Overall Similarity (%)", format="%.2f%%", width="medium"),
                            "Tag Similarity (%)": st.column_config.NumberColumn("Tag Similarity (%)", format="%.2f%%", width="medium"),
                            "Title Similarity (%)": st.column_config.NumberColumn("Title Similarity (%)", format="%.2f%%", width="medium"),
                            "Description Similarity (%)": st.column_config.NumberColumn("Description Similarity (%)", format="%.2f%%", width="medium"),
                            "Impressions": st.column_config.NumberColumn("Impressions", format="%d", width="medium"),
                            "CTR (%)": st.column_config.NumberColumn("CTR (%)", format="%.2f%%", width="small"),
                            "Views": st.column_config.NumberColumn("Views", format="%d", width="small"),
                            "Avg View Duration": st.column_config.TextColumn("Avg View Duration", width="medium"),
                            "Watch Time (hours)": st.column_config.NumberColumn("Watch Time (hours)", format="%.2f", width="medium"),
                            "Video Link": st.column_config.LinkColumn("Video Link", width="small"),
                            "Избранное": st.column_config.CheckboxColumn("Избранное", width="large"),
                        },
                        disabled=[
                            "Title", "Overall Similarity (%)", "Tag Similarity (%)", "Common Tags",
                            "Title Similarity (%)", "Common Title Words", "Description Similarity (%)",
                            "Common Description Words", "Impressions", "CTR (%)", "Views",
                            "Avg View Duration", "Watch Time (hours)", "Video Link"
                        ],
                        key="main_data_editor",
                        on_change=on_editor_change
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                    # --- Таблица "Избранное" ---
                    favorite_ids = [vid for vid, checked in st.session_state.favorites.items() if checked]
                    if favorite_ids:
                        st.subheader("Избранные видео")
                        # --- Исправление: убедиться, что столбец 'Избранное' есть и типа bool ---
                        if 'Избранное' in edited_df.columns:
                            # Преобразуем к bool (на случай object/float)
                            edited_df['Избранное'] = edited_df['Избранное'].astype(bool)
                            fav_df = edited_df[edited_df['Избранное']]
                        else:
                            fav_df = pd.DataFrame()
                        # Если fav_df пустой, но favorite_ids не пустой — показать предупреждение
                        if fav_df.empty and len(favorite_ids) > 0:
                            st.warning("Не удалось отобразить таблицу 'Избранные видео'. Проверьте, что столбец 'Избранное' корректно формируется.")
                        else:
                            st.markdown('<div class="table-container">', unsafe_allow_html=True)
                            st.dataframe(
                                fav_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Title": st.column_config.Column("Title", width="large"),
                                    "Overall Similarity (%)": st.column_config.NumberColumn("Overall Similarity (%)", format="%.2f%%", width="medium"),
                                    "Tag Similarity (%)": st.column_config.NumberColumn("Tag Similarity (%)", format="%.2f%%", width="medium"),
                                    "Title Similarity (%)": st.column_config.NumberColumn("Title Similarity (%)", format="%.2f%%", width="medium"),
                                    "Description Similarity (%)": st.column_config.NumberColumn("Description Similarity (%)", format="%.2f%%", width="medium"),
                                    "Impressions": st.column_config.NumberColumn("Impressions", format="%d", width="medium"),
                                    "CTR (%)": st.column_config.NumberColumn("CTR (%)", format="%.2f%%", width="small"),
                                    "Views": st.column_config.NumberColumn("Views", format="%d", width="small"),
                                    "Avg View Duration": st.column_config.TextColumn("Avg View Duration", width="medium"),
                                    "Watch Time (hours)": st.column_config.NumberColumn("Watch Time (hours)", format="%.2f", width="medium"),
                                    "Video Link": st.column_config.LinkColumn("Video Link", width="small"),
                                    "Избранное": st.column_config.CheckboxColumn("Избранное", width="large"),
                                }
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
