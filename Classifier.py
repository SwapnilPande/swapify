import argparse
import pprint
import sys
import os
import subprocess
import json
import spotipy
import spotipy.util as util
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from spotipy.oauth2 import SpotifyClientCredentials

import matplotlib.pyplot as plt

class Classifier:

    def __init__(self, client_id, client_secret) -> None:

        # Construct credentials manager
        self.client_credentials_manager = SpotifyClientCredentials(
            client_id = client_id,
            client_secret = client_secret
        )

        # Construct spotify client
        self.sp = spotipy.Spotify(client_credentials_manager = self.client_credentials_manager)

    def _get_features(self, username, playlist_id):
        offset = 0
        songs = []
        items = []
        ids = []

        # Get all songs
        while True:
            content = self.sp.user_playlist_tracks(username, playlist_id, fields=None, limit=100, offset=offset, market=None)
            songs += content['items']
            if content['next'] is not None:
                offset += 100
            else:
                break

        # Separate ids out
        for i in songs:
            ids.append(i['track']['id'])

        # Retrieve audio features for the song
        index = 0
        audio_features = []
        while index < len(ids):
            audio_features += self.sp.audio_features(ids[index:index + 50])
            index += 50

        # Convert dictionary of audio features in a list of features
        features_list = []
        for features in audio_features:
            features_list.append([features['energy'],
                                #   features['liveness'],
                                  features['tempo'],
                                #   features['speechiness'],
                                  features['acousticness'],
                                  features['instrumentalness'],
                                  #features['time_signature'],
                                  features['danceability'],
                                  #features['key'],
                                  features['loudness'],
                                  features['valence']])

        return ids, np.array(features_list)

    def _get_normalized_features(self, username, playlist_id):
        ids, features = self._get_features(username, playlist_id)

        # Get mean across songs for each features
        # We only need this for tempo, loudness
        # TODO handle time signature and key smartly
        means = np.mean(features, axis = 0)
        maxs = np.amax(features, axis = 0)

        # Scale tempo to [0,1]
        features[:,1] = features[:,1] / maxs[1]

        # Scale loudness to [0,1]
        #TODO this is a really hacky scaling, need to do this in a more principled way
        features[:,-2] = features[:,-2] * -1 /  np.amax(features[:,-2]*-1)

        return ids, features


    def get_user_playlist(self, username):
        playlists = self.sp.user_playlists(username)

        return {
            playlist['name']:playlist['id'] for playlist in playlists['items']
        }

    def get_playlist_songs(self, username, playlist_id):
        songs = []
        offset = 0
        while True:
            content = self.sp.user_playlist_tracks(username, playlist_id, fields=None, limit=100, offset=offset, market=None)
            songs += content['items']
            if content['next'] is not None:
                offset += 100
            else:
                break

        return {
            song['track']['id']:song['track']['name'] for song in songs
        }


    def _run_clustering(self, username, playlist_id):
        ids, features = self._get_normalized_features(username, playlist_id)

        # bgm = BayesianGaussianMixture(n_components=10, random_state=42).fit(features)
        # cluster_means = bgm.means_
        # cluster_assignments = bgm.predict(features)

        kmeans = KMeans(n_clusters=7).fit(features)

        cluster_assignments = kmeans.labels_
        cluster_means = kmeans.cluster_centers_

        # gm = GaussianMixture(n_components=10, random_state=0).fit(features)

        # return ids, gm.predict(features)

        return ids, cluster_assignments, cluster_means


if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)

    # Construct classifier object
    classifier = Classifier(config["client_id"], config["client_secret"])

    # Retrieve list of public playslists for specified user
    playlists = classifier.get_user_playlist(config["username"])

    # Get all songs from the playlist
    songs = classifier.get_playlist_songs(config["username"], config["playlist"])

    # Run clustering
    ids, features, cluster_centers = classifier._run_clustering(config["username"], config["playlist"])


    # Create bar graphs for each cluster
    # Create list of features
    # features = ['energy', 'tempo','acousticness', 'instrumentalness', 'danceability', 'loudness', 'valence']
    # for i in range(10):
    #     plt.figure()

    #     # Create bar for each cluster_center
    #     plt.bar(features, cluster_centers[i])

    #     plt.savefig('cluster_' + str(i) + '.png')


    # import ipdb; ipdb.set_trace()
    for i in range(10):
        indices = np.argwhere(features == i)
        print(f"------------------- Cluster {i} -------------------")
        for idx in indices:
            print(songs[ids[idx[0]]])
