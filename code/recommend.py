import h5reader
import random
from collections import deque

class Recommend:
	def __init__(self, playList1, playList2):
		track_list_id = getTrackList()
		self.All_songs = populateSongs(track_list_id)
		self.playList1 = playList1
		self.playList2 = playList2
		self.combined_playlist = {}
		self.current_song = None
		#NOTE: we could have time into song as a field 
		#but we can't play songs so it's a little difficult to use
		self.centroids1 = []
		self.centroids2 = []
		self.prob_1 = .5
		self.prob_2 = .5
		self.recent_songs = deque()
		self.new_cluster()

	def new_clusters(self):
		x = min(5,min(len(self.playList1),len(self.playList2)))
		for i in range(x):
			centroids1.append(self.playList1[random.choice(playList1.keys())])
			centroids2.append(self.playList2[random.choice(playList2.keys())])
		self.cluster()
	def cluster(self):
		self.cluster_playlist(1)
		self.cluster_playlist(2)

	def cluster_playlist(self, playlist):
		temp_playlist = dict(self.combined_playlist)
		added_playlist = {}
		centroids = []
		if(playList == 1):
			centroids = self.centroids1
			added_playlist = self.playList1
		else:
			centroids = self.centroids2
			added_playlist = self.playList2
		for key in added_playlist:
			temp_playlist[key] = temp_playlist.get(key, 0) + added_playlist[key]
		for i in range(20):

	def get_new_song(self):
		self.current_song = None

	def recommend(self, action):
		if(action == "play"): # liked the current song so set its value to be greater
			self.combined_playlist[self.current_song] = 
				self.combined_playlist.get(self.current_song, 0) + 1
		else(action == "skip"): #dislike the song so move on
			self.combined_playlist[self.current_song] = 
				self.combined_playlist.get(self.current_song, 0) - 1
		self.cluster() # changed some stuff so re-cluster everything 
		self.get_new_song() # get a new song 


