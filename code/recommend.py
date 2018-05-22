import h5reader
import random
from collections import deque

class Recommend:
	def __init__(self, playList1, playList2):

		#every single song that we have access to
		track_list_id = getTrackList()
		self.all_songs = populateSongs(track_list_id)

		#playlist of first person
		self.playList1 = playList1

		#playlist of second individual
		self.playList2 = playList2

		#songs that have been recommended
		self.combined_playlist = {}

		#current song
		self.current_song = None
		#NOTE: we could have time into song as a field 
		#but we can't play songs so it's a little difficult to use

		#centroids, plus probability of choosing each centroid to find song
		self.centroids1 = []
		self.centroid1_prob = []
		self.centroids2 = []
		self.centroid2_prob = []

		#probability of choosing a song using distribution 1, p_2 is 1 - p_1
		self.prob_1 = .5

		#queue of recent songs to avoid recommending several songs in a row
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
		self.get_new_song()

	# Call to cluster will update the two clusters formed from the different playlists
	def cluster(self):
		self.cluster_playlist(1)
		self.cluster_playlist(2)

	#Clusters the given playlist with the combined playlist as well
	#doesn't really work at the moment, need to account for number of times played
	def cluster_playlist(self, playlist):
		#start off temp playlist with just the combined playlist
		temp_playlist = dict(self.combined_playlist)
		added_playlist = {} #songs from original playlist
		centroids = [] #previous centroids for this part
		if(playList == 1): #first playlist
			centroids = self.centroids1
			added_playlist = self.playList1
		else: #second playlist
			centroids = self.centroids2
			added_playlist = self.playList2

		#add our original playlist to the combined one
		for key in added_playlist:
			temp_playlist[key] = temp_playlist.get(key, 0) + added_playlist[key]

		#cluster everything
		for i in range(20):
			#assign each song to a centroid
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

			#find centroid from assignments
			new_centroids = [{} for j in range(len(centroids))]
			for j in range(len(centroids)):
				new_centroids[j] = self.update_centroid(assignments[j])
			if(new_centroids == centroids):
				break
			centroids = new_centroids
			#set new values for centroid


	#helper function to update a single centroid based on the assigned songs
	def update_centroid(self, assigned):
		new_centroid = {}
		if(assigned == []):
			return new_centroid
		for song_id in assigned:
			song = self.all_songs[song_id]
			new_centroid['year']+=song.year
			
		return self.divide(new_centroid,len(assignedf))
	

	#helper function to divide every element by the total to get the average
	def divide(self, song, factor):
		for key in song:
			song[key] = song[key] / float(factor)
		return song

	#recommend a song
	def get_new_song(self):
		self.current_song = None

	#find distance between two songs
	def distance(self, centroid, song):
		return abs(centroid['year']-song.year)

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


