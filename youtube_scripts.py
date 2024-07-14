from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.tools import argparser
from dotenv import load_dotenv
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
# Set your API key
API_KEY = os.getenv('YOUTUBE_API_KEY', None)

if not API_KEY:
    raise AttributeError("YOUTUBE_API_KEY was not found in environment variables")

youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_video_comments(video_id: str) -> pd.DataFrame:
    
    comments = []
    nextPageToken = None
    while True:
        try:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,  # Adjust as needed
                pageToken=nextPageToken
            ).execute()

            # Extract comments
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(item)

            # Check if there are more comments
            nextPageToken = response.get('nextPageToken')
            if not nextPageToken:
                break

        except HttpError as e:
            print('An HTTP error %d occurred:\n%s' % (e.resp.status, e.content))
            break

    return pd.DataFrame([x['snippet']['topLevelComment']['snippet'] for x in comments])


def get_video_info(video_id):
    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    )
    response = request.execute()
    return response


def get_channel_info(channel_id):
    request = youtube.channels().list(
        part="snippet,contentDetails,statistics,brandingSettings",
        id=channel_id
    )
    response = request.execute()
    return response


