import json
import argparse
import spotipy
from ytmusicapi import YTMusic
from spotipy.oauth2 import SpotifyOAuth
from spotipy import Spotify
from tqdm import tqdm


class Migrator:
    
    def __init__(self,
                 args):

        sp_creds = json.load(open(args.spotify_credentials, 'r'))
        
        self.yt = YTMusic(args.youtube_music_credentials)
        self.sp = Spotify(auth_manager=SpotifyOAuth(client_id=sp_creds['client_id'],
                                                    client_secret=sp_creds['client_secret'],
                                                    redirect_uri="http://127.0.0.1:8000",
                                                    scope='user-library-modify'))
        
    def transfer_saved_tracks_yt_to_sp(self):
        
        saved_tracks = self.yt.get_liked_songs(limit = None)
                
        for track in tqdm(saved_tracks['tracks']):
            query_string = "remaster%20track:{}%20type:track" \
                           .format(track['title'])
            
            if track['album']:
                query_string += f"%20album:{track['album']['name']}"
                
            for artist in track['artists']:
                query_string += f"%20artist:{artist['name']}"
            
            result = self.sp.search(q=query_string, limit=1)
            if len(result['tracks'].get('items')) > 0:
                song_uri = result['tracks'].get('items')[0]['uri']
                self.sp.current_user_saved_tracks_add(tracks=[song_uri])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--spotify_credentials', metavar='S', type=str,
                        help='Provide the file location containing the spotify credentials')
    parser.add_argument('--youtube_music_credentials', metavar='Y', type=str,
                        help='Provide the file location containing the youtube music credentials')
    
    args = parser.parse_args()
    # print(args['youtube_music_credentials'])
    migration = Migrator(args)        
    migration.transfer_saved_tracks_yt_to_sp()