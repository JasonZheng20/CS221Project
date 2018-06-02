from h5reader import Song
import h5reader
import random
import h5py
import numpy as np
import time
import sys
import re
from collections import deque
import pickle


class Recommend:
	thetas = {
		'year':1.8,
   		'duration':1.3,
    	'key':1,
    	'generalLoudness':3,
    	'mode':1,
    	'tempo':3,
    	'timeSigniature':1,
    	'terms':10
	}

	def __init__(self, playList1, playList2):

		#every single song that we have access to
		self.all_songs = h5reader.load("../songsDict")
		self.all_song_centroids, self.all_song_assignments = h5reader.kMeansAllSongs(self.all_songs, 20, 100) 

		#playlist of first person
		self.playList1 = playList1

		#playlist of second individual
		self.playList2 = playList2

		#songs that have been recommended
		self.combined_playlist = {}

		#current song
		self.current_song = ""
		self.current_trackid = ""
		self.current_artist = ""
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
		for i in range(20):
			self.recent_songs.append("")
		self.new_cluster()

	# Initialize new clusters at the start. This function will be called at the beginning and have
	# 5 centroids by default
	def new_cluster(self):
		x = min(5,min(len(self.playList1),len(self.playList2)))
		key1 = np.random.choice(list(self.playList1),x,False)
		key2 = np.random.choice(list(self.playList2),x,False)
		for i in range(x):
			self.centroid1_prob.append(0.0)
			self.centroid2_prob.append(0.0)
			self.centroids1.append(self.all_songs[key1[i]])
			self.centroids2.append(self.all_songs[key2[i]])
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
		centroid_prob = []
		if(playlist == 1): #first playlist
			centroids = self.centroids1
			added_playlist = self.playList1
			centroid_prob = self.centroid1_prob
		else: #second playlist
			centroids = self.centroids2
			added_playlist = self.playList2
			centroid_prob = self.centroid2_prob

		#add our original playlist to the combined one
		for key in added_playlist:
			temp_playlist[key] = temp_playlist.get(key, 0) + 1

		#cluster everything
		for i in range(20):
			#reset the centroid probability
			for j in range(len(centroid_prob)):
				centroid_prob[j]=0.0
			#assign each song to a centroid
			assignments = [[] for j in range(len(centroids))]
			for key in temp_playlist:
				min_distance = -1
				min_centroid = 0
				for k in range(len(centroids)):
					distance = self.distance(centroids[k],self.all_songs[key])
					if(distance < min_distance or min_distance == -1):
						min_distance = distance
						min_centroid = k 
				assignments[min_centroid].append(key)
				centroid_prob[min_centroid]+=temp_playlist[key]


			#find centroid from assignments
			new_centroids = []
			for j in range(len(centroids)):
				new_centroids.append(self.update_centroid(assignments[j]))
			if(new_centroids == centroids):
				break
			centroids = new_centroids
			#set new values for centroid
		self.normalize_centroid_prob()


	#helper function to update a single centroid based on the assigned songs
	def update_centroid(self, assigned):
		new_centroid = Song("CENTROID")
		totals = {}
		if(assigned == []):
			return new_centroid
		for song_id in assigned:
			song = self.all_songs[song_id]

			new_centroid.year += song.year
			totals['year'] = totals.get('year',0) + (song.year>0)

        	new_centroid.duration += song.duration
           	totals['duration'] = totals.get('duration',0) + (song.duration>0)

           	new_centroid.key += song.key
           	totals['key'] = totals.get('key',0) + (song.key>0)

           	new_centroid.generalLoudness += song.generalLoudness
           	totals['generalLoudness'] = totals.get('generalLoudness',0) + (song.generalLoudness>0)

           	new_centroid.mode += song.mode
           	totals['mode'] = totals.get('mode',0) + (song.mode>0)

           	new_centroid.tempo += song.tempo
           	totals['tempo'] = totals.get('tempo',0) + (song.tempo>0)

           	new_centroid.timeSigniature += song.timeSigniature
           	totals['timeSigniature'] = totals.get('timeSigniature',0) + (song.timeSigniature>0)

           	new_centroid.terms = list(set(new_centroid.terms)+set(song.terms))
		return self.divide(new_centroid,totals)
	

	#helper function to divide every element by the total to get the average
	def divide(self, song, factors):
		song.year = song.year / float(max(1, factors['year']))
		song.duration = song.duration / float(max(1, factors['duration']))
		song.key = song.key / float(max(1, factors['key']))
		song.generalLoudness = song.generalLoudness / float(max(1, factors['generalLoudness']))
		song.mode = song.mode / float(max(1, factors['mode']))
		song.tempo = song.tempo / float(max(1, factors['tempo']))
		song.timeSigniature = song.timeSigniature / float(max(1, factors['timeSigniature']))
		return song

	#recommend a song
	def get_new_song(self):
		centroids = []
		centroid_prob = []
		if(random.random() > self.prob_1):
			self.prob_1 +=.05
			centroids = self.centroids2
			centroid_prob = self.centroid2_prob
		else:
			self.prob_1 -=.05
			centroids = self.centroids1
			centroid_prob = self.centroid1_prob
		choice = random.random()
		index = 0
		running_total = 0.0
		while index < len(centroids):
			running_total += centroid_prob[index]
			if(choice < running_total):
				break
			index+=1
		chosen_centroid = centroids[index]
		min_distance = float('inf')
		min_index = 0
		for i in range(len(self.all_song_centroids)):
			distance = self.distance(self.all_song_centroids[i], chosen_centroid)
			if(distance < min_distance):
				min_distance=distance
				min_index = i
		min_distance = float('inf')
		min_song = ""
		for song in self.all_song_assignments[min_index]:
			distance = self.distance(song, chosen_centroid)
			trackid = song.trackid
			if(distance < min_distance and not trackid in self.recent_songs):
				min_distance=distance
				min_song = song

		self.current_song = min_song.title
		self.current_artist = min_song.artistName
		self.current_trackid = min_song.trackid
		self.recent_songs.popleft()
		self.recent_songs.append(min_song.trackid)

	#find distance between two songs
	def distance(self, centroid, song):
		distance =  self.thetas['year']*abs(centroid.year - song.year)
		distance += self.thetas['duration']*abs(centroid.duration - song.duration)
		distance += self.thetas['key']*abs(centroid.key - song.key)
		distance += self.thetas['generalLoudness']*abs(centroid.generalLoudness - song.generalLoudness)
		distance += self.thetas['mode']*abs(centroid.mode - song.mode)
		distance += self.thetas['tempo']*abs(centroid.tempo - song.tempo)
		distance += self.thetas['timeSigniature']*abs(centroid.timeSigniature - song.timeSigniature)
		distance += self.thetas['terms']*self.term_distance(centroid, song)
		return distance

	#distance between the terms of the two 
	def term_distance(self, centroid, song):
		cent_terms = set(centroid.terms)
		song_terms = set(song.terms)
		unified = cent_terms.intersection(song_terms)
		return (len(unified)+1)/float(len(song_terms)+1)


	#actual call, to try to recommend a song after you play or skip a song
	def recommend(self, action):
		if(action == "play"): # liked the current song so set its value to be greater
			self.combined_playlist[self.current_trackid] = \
				self.combined_playlist.get(self.current_trackid, 0) + 1
		elif(action == "skip"): #dislike the song so move on
			self.combined_playlist[self.current_trackid] = \
			self.combined_playlist.get(self.current_trackid, 0) - 1
		else:
			return
		self.cluster() # changed some stuff so re-cluster everything 
		self.get_new_song() # get a new song 

	def normalize_centroid_prob(self):
		total1 = 0.0
		total2 = 0.0
		for i in range(len(self.centroid1_prob)):
			self.centroid1_prob[i] = 1.0/(1+2**(-1*self.centroid1_prob[i]))
			self.centroid2_prob[i] = 1.0/(1+2**(-1*self.centroid2_prob[i]))
			total1 += self.centroid1_prob[i]
			total2 += self.centroid2_prob[i]
		for i in range (len(self.centroid1_prob)):
			self.centroid1_prob[i] = self.centroid1_prob[i]/total1
			self.centroid2_prob[i] = self.centroid2_prob[i]/total2



#RECOMMENDER script
playList1 = []
playList2 = []
play = open("../data/playlist1.txt")
for line in play:
	playList1.append(line.rstrip())
play = open("../data/playlist2.txt")
for line in play:
	playList2.append(line.rstrip())
recommend = Recommend(playList1,playList2)
current_song = recommend.current_song
current_artist = recommended.current_artist
while(True):
	choice = raw_input("Your current song is \"" + current_song + "\" by " + current_artist + "would you like to play or skip (enter quit to exit) ").lower()
	if(choice == "quit"):
		break
	elif(choice == "play" or choice == "skip"):
		recommend.recommend(choice)
		current_song = recommend.current_song
		current_artist = recommended.current_artist
	else:
		print("I'm sorry, you need to enter 'play' 'skip' or 'quit'")










