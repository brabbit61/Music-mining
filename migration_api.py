import json
import time
import argparse
import spotipy
from ytmusicapi import YTMusic
from spotipy.oauth2 import SpotifyOAuth
from spotipy import Spotify
from tqdm import tqdm


class User:

    def __init__(self,
                 args):

        self.sp_creds = json.load(open(args.spotify_credentials, 'r'))
        self.yt = YTMusic(args.youtube_music_credentials)
        self.sp = None

    def set_client_scope(self, scope):
        self.sp = Spotify(
            auth_manager=SpotifyOAuth(
                client_id=self.sp_creds['client_id'],
                client_secret=self.sp_creds['client_secret'],
                redirect_uri="http://127.0.0.1:8000",
                scope=scope))

    def transfer_saved_tracks_yt_to_sp(self):

        sp_saved_tracks = []
        offset = 0

        self.set_client_scope("user-library-read")

        yt_saved_tracks = self.yt.get_liked_songs(limit=None)
        tracks = self.sp.current_user_saved_tracks(limit=None).get('items')

        while tracks != []:
            sp_saved_tracks.extend([track['track']['name']
                                   for track in tracks])
            offset += len(tracks)
            tracks = self.sp.current_user_saved_tracks(
                limit=None, offset=offset).get('items')

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
    # print(args['youtube_music_credentials'])
    user = User(args)
    user.transfer_saved_tracks_yt_to_sp()
