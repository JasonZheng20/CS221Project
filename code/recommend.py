import h5reader
import random
from collections import deque
import math

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

	# Initialize new clusters at the start. This function will be called at the beginning and have
	# 5 centroids by default
	def new_clusters(self):
		x = min(5,min(len(self.playList1),len(self.playList2)))
		for i in range(x):
			centroids1.append(self.playList1[random.choice(playList1.keys())])
			centroids2.append(self.playList2[random.choice(playList2.keys())])
		self.cluster()

	# Call to cluster will update the two clusters formed from the different playlists
	def cluster(self):
		self.cluster_playlist(1)
		self.cluster_playlist(2)

	#Clusters the given playlist with the combined playlist as well
	#doesn't really work at the moment, need to account for number of times played
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
			assignments = [[] for j in range(len(centroids))]
			for j in range(len(temp_playlist)):
				min_distance = -1
				min_centroid = 0
				for k in range(len(centroids)):
					distance = self.dist(centroids[k],temp_playlist[j])
					if(distance < min_distance or min_distance == -1):
						min_distance = distance
						min_centroid = k 
				assignments[min_centroid].append(temp_playlist[j])
			#assign each song to a centroid
			new_centroids = [[] for j in range(len(centroids))]
			totals = [len(assignments[j]) for j in range(len(centroids))]
			for j in range(len(centroids)):
				for k in range(totals[j]):
					if(k == 0):
						new_centroids[j]=assignments[j][k]
					else:
						new_centroids[j] = self.add(new_centroids[j],assignments[j][k])
				new_centroids[j] = self.divide(new_centroids[j],totals[j])
			if(new_centroids == centroids):
				break
			centroids = new_centroids
			#set new values for centroid

	#helper function to divide every element by the total to get the average
	def divide(self, song, factor):
		for key in song:
			song[key] = song[key] / float(factor)
		return song

	#adds up the two songs
	def add(self, song1, song2):
		total = {}
		for key in (set(song1.keys()) + set(song2.keys())):
			total[key] = song1.get(key, 0) + song2.get(key, 0)
		return total

	#recommend a song
	def get_new_song(self):
		self.current_song = None

	#find distance between two songs
	def distance(self, song1, song2):
		return math.abs(song1['year']-song2['year'])

	#actual call, to try to recommend a song after you play or skip a song
	def recommend(self, action):
		if(action == "play"): # liked the current song so set its value to be greater
			self.combined_playlist[self.current_song] = 
				self.combined_playlist.get(self.current_song, 0) + 1
		elif(action == "skip"): #dislike the song so move on
			self.combined_playlist[self.current_song] = 
				self.combined_playlist.get(self.current_song, 0) - 1
		else:
			return
		self.cluster() # changed some stuff so re-cluster everything 
		self.get_new_song() # get a new song 


