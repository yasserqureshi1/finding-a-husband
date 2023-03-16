import numpy as np


def remove_low_length_tracks(tracks, min_length=10):
    '''
    Remove tracks that have a length less than `min_length`

    Params:
     - tracks (arr): Trial array
     - min_length (int): Minimum length for track

    Returns:
     - tracks (arr): Trial array with low length tracks removed
    '''
    new_tracks = []
    for track in tracks:
        new_mossies = []
        for mossie in track:
            if len(mossie) > min_length:
                new_mossies.append(mossie)
        new_tracks.append(new_mossies)

    return new_tracks


def velocity(track):
    '''
    Generates velocity features from track

    Params:
     - track (list or NumPy array)

    Returns:
     - x_dot (NumPy array)
     - y_dot (NumPy array)
     - z_dot (NumPy array)
    '''
    track = np.array(track) if type(track) != np.ndarray else track
    x_dot = np.abs((track[2:, 0] - track[:-2, 0])/(2/25))
    y_dot = np.abs((track[2:, 1] - track[:-2, 1])/(2/25))
    z_dot = np.abs((track[2:, 2] - track[:-2, 2])/(2/25))
    return x_dot, y_dot, z_dot


def append_vel(track):
    '''
    Appends velocity features to track

    Params:
     - track (list or NumPy array)

    Returns:
     - track (NumPy array)
    '''
    x_dot, y_dot, z_dot = velocity(track)
    track = np.insert(track, len(track[0]), np.append(x_dot, [np.nan,np.nan]), axis=1)
    track = np.insert(track, len(track[0]), np.append(y_dot, [np.nan,np.nan]), axis=1)
    track = np.insert(track, len(track[0]), np.append(z_dot, [np.nan,np.nan]), axis=1)
    return track
