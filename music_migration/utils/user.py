import json
import time
import argparse
import math
import pandas as pd
import numpy as np
from collections import Counter
from ytmusicapi import YTMusic
from spotipy.oauth2 import SpotifyOAuth
from spotipy import Spotify
from tqdm import tqdm
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from scipy.spatial import distance
from .helper import *

class User:
    # TODO: Add function to migrate playlists as well
    
    def __init__(self, args):

        self.sp_creds = json.load(open(args["spotify_credentials"], 'r'))
        self.yt = YTMusic(args["youtube_music_credentials"])
        self.sp = None

    def set_client_scope(self, scope):
        """
        Instantiate an object with an appropriate scope to access Spotify data using a client ID, client secret and redirect URI.

        Parameters:
        - scope (str): specifies the scope for calling Spotify API to minimize the access to user data

        Returns:
          None
        """

        self.sp = Spotify(
            auth_manager=SpotifyOAuth(
                client_id=self.sp_creds['client_id'],
                client_secret=self.sp_creds['client_secret'],
                redirect_uri="http://127.0.0.1:8000",
                scope=scope))

    def get_yt_saved_songs(self):
        return self.yt.get_liked_songs(limit=None)

    def get_yt_playlists(self):

        saved_playlists = self.yt.get_library_playlists(limit=None)
        all_playlists = {}
        
        for playlist in saved_playlists:
            
            if playlist['title'] == "Your Likes":
                continue
            
            all_playlists[playlist['title']] = []
            playlist_data = self.yt.get_playlist(playlistId=playlist['playlistId'], limit=None)
            tracks = playlist_data['tracks']
            
            for track in tracks:
                track_data = {}
                
                track_data['track_name'] = track['title']
                
                if track['album']:
                    track_data['album'] = track['album']['name']

                track_data['artists'] = []
                for artist in track['artists']:
                    track_data['artists'].append(artist['name'])
                
                all_playlists[playlist['title']].append(track_data)
                    
        return all_playlists
             
    def get_sp_saved_songs(self):
        saved_tracks = []
        offset = 0
        self.set_client_scope("user-library-read")

        tracks = self.sp.current_user_saved_tracks(limit=None).get('items')

        while tracks != []:
            saved_tracks.extend(tracks)
            offset += len(tracks)
            tracks = self.sp.current_user_saved_tracks(
                limit=None, offset=offset).get('items')

        return saved_tracks
    
    def get_sp_playlist_tracks(self, playlist_uri):
        saved_tracks = []
        offset = 0
        self.set_client_scope("playlist-read-private,playlist-read-collaborative")

        tracks = self.sp.user_playlist_tracks(limit=None, 
                                              playlist_id=playlist_uri).get('items')
        
        while tracks != []:
            saved_tracks.extend(tracks)
            offset += len(tracks)
            tracks = self.sp.user_playlist_tracks(limit=None, 
                                                  playlist_id=playlist_uri,
                                                  offset=offset).get('items')

        return saved_tracks
    
    def get_sp_playlists(self):
        all_playlists = []
        offset = 0
        self.set_client_scope("playlist-read-private,playlist-read-collaborative")
        
        playlists = self.sp.current_user_playlists(limit=None).get('items')

        while playlists != []:
            all_playlists.extend(playlists)
            offset += len(playlists)
            playlists = self.sp.current_user_playlists(
                limit=None,
                offset=offset).get('items')

        return all_playlists

    def transfer_saved_tracks_yt_to_sp(self):
        
        sp_saved_tracks = self.get_sp_saved_songs()
        yt_saved_tracks = self.get_yt_saved_songs()

        add_songs = [song for song in yt_saved_tracks.get('tracks')
                     if song['title'] not in sp_saved_tracks]

        add_songs = list(set([song['title'] for song in yt_saved_tracks.get('tracks')]) -
                         set(sp_saved_tracks))
        add_songs = [song for song in yt_saved_tracks.get('tracks')
                     if song['title'] in add_songs]

        self.set_client_scope("user-library-modify")

        for index, track in tqdm(enumerate(add_songs)):
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

            if index % 200 == 0:
                time.sleep(1)

    def transfer_saved_playlists_yt_to_sp(self):
        yt_saved_playlists = self.get_yt_playlists()
        self.set_client_scope("playlist-modify-private")

        for playlist_name, tracks in yt_saved_playlists.items():
            tracks_uri = []
            for track in tracks:
            
                query_string = "remaster%20track:{}%20type:track" \
                            .format(track['track_name'])

                if "album" in yt_saved_playlists[playlist_name]:
                    query_string += f"%20album:{track['album']}"

                for artist in track['artists']:
                    query_string += f"%20artist:{artist}"

                result = self.sp.search(q=query_string, limit=1)

                if len(result['tracks'].get('items')) > 0:
                    tracks_uri.append(result['tracks'].get('items')[0]['uri'])
            
            self.create_playlist(playlist_name, tracks_uri)

    def generate_playlists(self, num_playlists=None, num_songs=15):
        
        # for spotify
        saved_tracks = self.get_sp_saved_songs()
        
        if len(saved_tracks) == 0:
            return
        
        # TODO:  Add Genre information for each track below so that it can be used during clustering
        audio_features = pd.DataFrame(self.sp.audio_features(
            tracks=[saved_tracks[0]['track']['uri']]))
        
        for i in range(1, len(saved_tracks)):
            audio_features = pd.concat([audio_features,
                                        pd.DataFrame(
                                            self.sp.audio_features(tracks=[saved_tracks[i]['track']['uri']]))])

        names = pd.DataFrame({
            "uri": [song['track']['uri'] for song in saved_tracks],
            "name": [song['track']['name'] for song in saved_tracks]
        })
        audio_features = audio_features.merge(names, how='left', on="uri")

        audio_features.drop(columns=["type",
                                     "track_href",
                                     "analysis_url"], inplace=True)

        col_transformer = make_column_transformer(
            (StandardScaler(), ['danceability',
                                'energy',
                                'loudness',
                                'speechiness',
                                'acousticness',
                                'instrumentalness',
                                'liveness',
                                'valence',
                                'tempo',
                                'duration_ms']),
            (OneHotEncoder(drop="if_binary", handle_unknown="ignore"), ["mode",
                                                                        "key",
                                                                        "time_signature"]),
            remainder="passthrough"
        )

        transformed_audio_features = pd.DataFrame(
            col_transformer.fit_transform(audio_features),
            columns=col_transformer.get_feature_names_out())

        if num_playlists is None:
            # If number of playlists have not been defined, get the best number of clusters
            num_playlists, _ = get_num_clusters(transformed_audio_features.drop(["remainder__name",
                                                                                 "remainder__uri",
                                                                                 "remainder__id"],
                                                                                axis=1))
            
        km = KMeans(
            n_clusters=num_playlists, init='random',
            n_init=10, max_iter=5000,
            tol=1e-04, random_state=42
        )
        transformed_audio_features['cluster'] = km.fit_predict(transformed_audio_features.drop(["remainder__name",
                                                                                                "remainder__uri",
                                                                                                "remainder__id"],
                                                                                               axis=1))
        transformed_audio_features['cluster'] = transformed_audio_features['cluster'].astype(np.int64)
        
        centers = km.cluster_centers_
        print(transformed_audio_features.head())
        transformed_audio_features["distance"] = transformed_audio_features.apply(
            get_cosine_distance, centers=centers, axis=1)

        top_songs = transformed_audio_features.sort_values(['distance']) \
            .groupby("cluster") \
            .head(num_songs) \
            .reset_index(drop=True)
        
        for n in range(num_playlists):
            self.create_playlist(f"Auto-Gen-Playlist-{n}", 
                                 top_songs[top_songs['cluster'] == n]["remainder__uri"].tolist())

    def create_playlist(self, playlist_name, tracks_ids):
        
        self.set_client_scope("user-library-read,playlist-modify-private,playlist-modify-public")
        playlist_description = ''

        # Create the playlist
        playlist = self.sp.user_playlist_create(self.sp.current_user()['id'],
                                                playlist_name,
                                                description=playlist_description)
        
        # Maximum songs you can add in one call = 100
        for i in range(math.ceil(len(tracks_ids)/100)):
            self.sp.playlist_add_items(playlist['id'], tracks_ids[i*100 : (i+1)*100])
        
    def create_similar_playlist(self, playlist_name):
        self.set_client_scope('user-read-private')
        country_code = self.sp.me()['country']
        self.set_client_scope("playlist-read-private,playlist-read-collaborative,playlist-modify-private,playlist-modify-public")
        playlists = self.get_sp_playlists()
        tracks = None
        
        for playlist in playlists:
            if playlist['name'] == playlist_name:
                tracks = self.get_sp_playlist_tracks(playlist_uri=playlist['uri'])
                break
        
        if tracks == None:
            return
        
        # seed_tracks = ','.join([track['track']['uri'] for track in tracks])
        
        artists = Counter()
        genres = Counter()
        
        for track in tracks:
            track_data = self.sp.track(track_id=track['track']['uri'])
            
            for artist in track_data['artists']:
                artists[artist['uri']] += 1
                if artist.get('genres'):
                    for genre in artist.get('genres'):
                        genres[genre] += 1
        
        # Max number of seeds allowed is 5 (inclusive of all: tracks, artists and genres)
        
        # seed_artists = ','.join(sorted(artists, key=artists.get, reverse=True)[:5])
        # seed_genres = ','.join(sorted(artists, key=artists.get, reverse=True)[:5])
        seed_artists = sorted(artists, key=artists.get, reverse=True)[:2]
        seed_genres = sorted(genres, key=genres.get, reverse=True)[:2]
        seed_tracks = [track['track']['uri'] for track in tracks[:5-len(seed_artists)-len(seed_genres)]]
        
        recommended_tracks = self.sp.recommendations(seed_artists=seed_artists,
                                seed_genres=seed_genres,
                                seed_tracks=seed_tracks,
                                limit=20,
                                country=country_code)
        
        tracks_uri = [track['uri'] for track in recommended_tracks['tracks']]
        self.create_playlist(f"Similar to {playlist_name}", tracks_uri)
        
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--spotify_credentials',
        metavar='S',
        type=str,
        help='Provide the file location containing the spotify credentials')
    parser.add_argument(
        '--youtube_music_credentials',
        metavar='Y',
        type=str,
        help='Provide the file location containing the youtube music credentials')
    
    args = parser.parse_args()
    user = User(args)
    user.generate_playlists(num_songs=35)
