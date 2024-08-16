import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import concurrent.futures
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import re
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from functools import lru_cache
import asyncio
import aiohttp
from language_util import extract_language  # Import the utility function

# Load environment variables
load_dotenv()

# Spotify credentials
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI')

# Scopes define the access level to a user's data
scope = "user-library-read playlist-read-private"

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                               client_secret=SPOTIPY_CLIENT_SECRET,
                                               redirect_uri=SPOTIPY_REDIRECT_URI,
                                               scope=scope))

# Initialize ChatOpenAI with your API key
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=openai_api_key, temperature=0.5, model_name="gpt-4")

# Advanced text preprocessing function
def preprocess_text_advanced(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[\u2600-\u26FF\u2700-\u27BF]', '', text)  # Remove emojis and symbols
    return text.strip()

# Function to detect language using OpenAI via LangChain with caching
@lru_cache(maxsize=1000)
def detect_language_openai(track_name, artist_name):
    prompt_template = PromptTemplate(
        input_variables=["track_name", "artist_name"],
        template="Track: {track_name}\nArtist: {artist_name}\nWhat is the language of this song?"
    )
    
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    response = llm_chain.run({"track_name": track_name, "artist_name": artist_name})
    
    # Clean and extract the language using the utility function
    language = extract_language(response)
    return language

# Asynchronous function to retrieve playlist tracks
async def get_playlist_tracks_async(playlist_id):
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

# Function to group tracks by language using concurrent.futures
def group_tracks_by_language(tracks, detect_language_func):
    grouped_tracks = defaultdict(list)
    
    def process_track(track):
        track_name = preprocess_text_advanced(track['track']['name'])
        artist_name = preprocess_text_advanced(track['track']['artists'][0]['name'])
        language = detect_language_func(track_name, artist_name)
        if language:
            return language, track
        return None, None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        futures = [executor.submit(process_track, track) for track in tracks]
        for future in concurrent.futures.as_completed(futures):
            try:
                language, track = future.result()
                if language and track:
                    grouped_tracks[language].append(track)
            except Exception as e:
                print(f"Error processing track: {e}")
    
    return grouped_tracks

# Function to generate the report and count tracks by language
def generate_language_report(grouped_tracks):
    report = []
    language_count = {}
    
    for language, tracks in grouped_tracks.items():
        language_count[language] = len(tracks)
        
        report.append(f"The language of this song is {language}.")
        report.append(f"Total Tracks: {len(tracks)}")
        
        for track in tracks:
            track_name = track['track']['name']
            artist_name = track['track']['artists'][0]['name']
            report.append(f"{track_name} by {artist_name}")
        
        report.append("")
    
    if report and report[-1] == "":
        report.pop()
    
    return "\n".join(report), language_count

# Visualization function
def plot_language_distribution(language_count):
    languages = list(language_count.keys())
    counts = list(language_count.values())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=languages, palette="viridis")
    plt.title("Number of Tracks per Language in Playlist")
    plt.xlabel("Number of Tracks")
    plt.ylabel("Language")
    plt.show()

async def main(playlist_id):
    tracks = await get_playlist_tracks_async(playlist_id)
    grouped_tracks = group_tracks_by_language(tracks, detect_language_openai)
    
    report, language_count = generate_language_report(grouped_tracks)
    print(report)
    
    plot_language_distribution(language_count)

# Replace with your correct playlist ID
playlist_id = '37i9dQZF1EVKuMoAJjoTIw'

# Run the async main function
asyncio.run(main(playlist_id))
