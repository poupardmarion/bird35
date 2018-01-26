
# coding: utf-8

# In[1]:


import logging
import wave
import os
import glob
import json
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import timezonefinder
import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')

logging.basicConfig(level=logging.INFO)


# # Create the dataset
#
# Extract `bird_marion.zip` in the current directory.
#
# The following cell will perform the following tasks:
# - Parse the JSONs
# - Transcode/resample the MP3s to WAVs
# - Create a Pandas DataFrame and serialize it
#
# If the serialized dataframe already exists all the tasks will be skipped and the file loaded.

# In[2]:


jsons_dir = 'jsons'
download_dir = 'downloads'
dataset_dir = 'dataset'

audio_dir = f'{dataset_dir}/audio'
dataset_filename = f'{dataset_dir}/dataset.h5'

if not os.path.isfile(dataset_filename):
    logging.info(f'Prepare dataset and write to {dataset_filename}')

    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    recordings = []
    for json_file in glob.glob(f'{jsons_dir}/*.json'):
        with open(json_file) as f:
            d = json.load(f)
            recordings += d['recordings']

    df = pd.DataFrame(recordings)

    def find_filename(id_value):
        pattern = f'{download_dir}/{id_value}.*'
        return glob.glob(pattern)[0]

    df['original_filename'] = df['id'].map(lambda x: find_filename(x))
    df['wav_filename'] = df['id'].map(lambda x: f'{dataset_dir}/audio/{x}.wav')

    logging.info('Start transcoding...')

    progress = tqdm.tqdm(df.itertuples())
    for row in progress:
        if os.path.exists(row.wav_filename):
            continue

        cmd = f"sox {row.original_filename} {row.wav_filename} rate 44100 channels 1"
        progress.set_description(f'Command {cmd}')
        subprocess.call(cmd, shell=True)

    logging.info('Finished transcoding...')

    df['lat'] = df['lat'].astype(float)
    df['lng'] = df['lng'].astype(float)
    df['cnt'] = df['cnt'].astype('category')
    df['en'] = df['en'].astype('category')
    df['gen'] = df['gen'].astype('category')
    df['sp'] = df['sp'].astype('category')
    df['ssp'] = df['ssp'].astype('category')
    df['rec'] = df['rec'].astype('category')

    df.to_hdf(dataset_filename, 'dataset', format='table')

else:
    logging.info(f'Read dataset from {dataset_filename}')

    df = pd.read_hdf(dataset_filename)


# # Clean up data
#
# ## Remove data without geolocation info

# In[3]:


# Remove data without Geo info
with_geo = (df['lat'].notnull() & df['lng'].notnull())
count_all = len(df)
df = df[with_geo]
count_geo = len(df)

num_removed = count_all - count_geo
logging.warning(f'Removed {num_removed} data points that lack geolocation.')


# ## Parse timestamp information

# In[4]:


df.head()[['date', 'time']]


# In[7]:


tf = timezonefinder.TimezoneFinder()
df['timezone'] = df.apply(lambda row: tf.timezone_at(lng=row['lng'], lat=row['lat']), axis=1)

df.head()[['cnt', 'timezone']]


# In[8]:


def timestamp_original(r):
    try:
        return pd.Timestamp(f"{r['date']} {r['time']}", tz=None)
    except ValueError:
        return None

df['timestamp_local'] = df.apply(timestamp_original, axis=1)
df['timestamp_utc'] = df.apply(lambda x: x['timestamp_local'].tz_localize(x['timezone']).tz_convert('utc'), axis=1)


# ## Parse audio information

# In[9]:


def get_wav_length(wav_filename):
    with wave.open(wav_filename, 'rb') as wav_file:
        return wav_file.getnframes()

df['wav_nframes'] = df['wav_filename'].apply(get_wav_length)

fnm = df['wav_filename'][0]
with wave.open(fnm, 'rb') as f:
    samplerate = f.getframerate()

df['wav_length'] = df['wav_nframes'] / float(samplerate)


# ## Define the class of each recording

# In[10]:


df['class'] = df['en']
df[['class']].drop_duplicates()


# In[11]:


with pd.option_context('display.max_rows', None):
    display(df[['class', 'gen', 'sp', 'ssp']].drop_duplicates())


# # Analyze the dataset
#
# ## Global analysis
#
# We first show the total amount of classes.

# In[12]:


print(f"Unique classes: {df['class'].nunique()}")


# In[13]:


print(f"Total wav length: {df['wav_length'].sum() / 3600.0:.2f}h")


# In[14]:


print(f"Unique recorders: {df['rec'].nunique()}")


# In[15]:


print("Audio length per class:")
with pd.option_context('display.precision', 0):
    display(df.groupby(['class'])[['wav_length']].sum().sort_values('wav_length'))


# In[16]:


print("Audio length stats per class:")
with pd.option_context('precision', 1):
    display(df.groupby(['class'])['wav_length'].describe().sort_values(by=['count']))


# In[17]:


print("Number of different recorders per class")
df.groupby(['class'])[['rec']].nunique().sort_values('rec')


# In[18]:


import folium
import folium.plugins

m = folium.Map([48., 5.],
               tiles='stamentoner',
               zoom_start=5)

# Plot heatmap
data = df[['lat', 'lng']].values.tolist()
folium.plugins.HeatMap(data).add_to(m)

# Plot point cloud
def plot_point(row):
    folium.CircleMarker(location=[row['lat'], row['lng']],
                        radius=2,
                        weight=2,
                        stroke=False,
                        color='#ffffff',
                        fill=True,
                        fill_color='#333333',
                        fill_opacity=1,
                        opacity=1,
                        ).add_to(m)

df.apply(plot_point, axis=1)

m


# ### Time of the day of recordings

# In[20]:


df['hour_of_day'] = (df['timestamp_local'] - df['timestamp_local'].dt.normalize()) / np.timedelta64(1, 'h')
sns.distplot(df['hour_of_day'].dropna())


# In[21]:


df.groupby('class')[['wav_length']].hist(bins=24)


# In[68]:


g = sns.FacetGrid(df, col="class", col_wrap=4, margin_titles=True, sharex=True)
bins = np.linspace(0, 500, 30)
g.map(sns.distplot, "wav_length", color="steelblue", bins=bins)
g.set(xlim=(0, 500))
