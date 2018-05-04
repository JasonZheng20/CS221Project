import random
import math

def logistic(x):
	return 1.0/(1+math.exp(-x))

class BasicRecommender():
	def __init__(self,data_file,user_playlist):
		self.artistMap = {}
		self.songs = {}
		self.userPrefs = {}
		self.currentSong = ""
		data = open(data_file, "r")
		for line in data:
			pieces = line.split(",")
			song = pieces[0]
			artist = pieces[1]
			self.songs[song] = artist
			self.artistMap[artist] = self.artistMap.get(artist, []) + [song]
		playlist = open(user_playlist, "r")
		for line in playlist:
			artist = self.songs.get(line, None)
			if(artist == None):
				continue
			self.userPrefs[artist] = self.userPrefs.get(artist, 0) + 1

	def nextSong(self):
		max_value = 0
		chosen_artist = ""
		for artist in self.artistMap:
			value = logistic(self.userPrefs.get(artist,0)) + random.random()
			if(value > max_value):
				max_value = value
				chosen_artist = artist
		possible_songs = self.artistMap[chosen_artist]
		self.currentSong = random.choice(possible_songs)

	def play(self):
		currArtist = self.songs[self.currentSong]
		self.userPrefs[currArtist] = self.userPrefs.get(currArtist, 0) + 1

	def skip(self):
		currArtist = self.songs[self.currentSong]
		self.userPrefs[currArtist] = self.userPrefs.get(currArtist, 0) - 1

basicRec = BasicRecommender("../data/data.txt", "../data/playlist.txt")
basicRec.nextSong()
currSong =  basicRec.currentSong
while(True):
	choice = raw_input("Your current song is \"" + currSong + "\" would you like to play or skip (enter quit to exit) ").lower()
	if(choice == "quit"):
		break
	elif(choice == "play"):
		basicRec.play()
		basicRec.nextSong()
		currSong = basicRec.currentSong
	elif(choice == "skip"):
		basicRec.skip()
		basicRec.nextSong()
		currSong = basicRec.currentSong
	else:
		print("I'm sorry, you need to enter 'play' 'skip' or 'quit'")



