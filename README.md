# swapify
A package of extensions to spotify to give you the music you want to listen to


## Setup
* Make a new account and app at [developer.spotify.com](developer.spotify.com)
* Code depends on `spotipy` and `sklearn`, along with the standard set of python packages. Install these with pip
* Make a copy of `samples_config.json`, called `config.json`. Populate the fields with your `client_id` and `client_secret` from the spotify developer dashboard, and your `username` for your spotify account.
* `Classifier.py` is the entry point to the code, you can the `get_user_playlist` function to retrieve the playlist_id for the playlist on which you want to run clustering.
