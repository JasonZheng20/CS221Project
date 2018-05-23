#-------------------------------------------------------------------------------
#h5reader.py

#IMPORTANT: PLACE THIS FILE IN THE ORIGIN FOLDER
#This file contains code to load and save Song objects from song datasets
#-------------------------------------------------------------------------------
import h5py
import numpy as np
import time
import sys
import re
import pickle
import random

#-------------------------------------------------------------------------------
# favoriteSong(cluster)

#gets the lowest distance song to a centroid
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# randomCentroids(songsDict, numCentroids)

#Takes songsDict, and constructs numCentroids # of random centroids
#-------------------------------------------------------------------------------
def randomCentroids(songsDict, numCentroids):
    centroids = []
    for i in xrange(numCentroids):
        centroids.append(songsDict[random.choice(songsDict.keys())]) #currently song obj
    return centroids

#-------------------------------------------------------------------------------
#distanceSongs(song1, song2)

#Takes two song objects and computes their distance
#Currently computes distance solely by year
#Distance metrics to add:
#
#1. Artist similarity
#2. Same Album ( learn weights for that? )
#3. Duration of song
#4. Genre of song
#5. terms (a bunch of strings, probably booleans again)

#Extended metrics:
#6. Max volume
#7. Pitches (?)
#8. Research dataset more
#-------------------------------------------------------------------------------
def distanceSongs(song1, song2):
    #TODO DO A CHECK TO SEE IF YEAR (AND OTHER DATA FIELDS ARE UNINIT (0))
    return abs(song1.year - song2.year)

#-------------------------------------------------------------------------------
#kMeansAllSongs(songsDict, numCentroids)

#Takes a dictionary of trackid:Song object, and creates a clustering of similar
#songs using K-means clustering algorithm.
#currently storing each thing as the song object itself, due to computation ease
#returns centroids and clusterings

#TODO: update kMeans so that it works on an arbitrary number of fields, as described above
#-------------------------------------------------------------------------------
def kMeansAllSongs(songsDict, numCentroids = 5, T = 100000):
    centroids = randomCentroids(songsDict, numCentroids)
    for i in xrange(0, T):
        assignments = [[] for j in range(len(centroids))] #each subarray is a cluster
        for song in songsDict:
            min_distance = float('inf')
            min_centroid = 0
            for k in xrange(0, len(centroids)):
                if centroids[k] == CONST_FILLER_SONG: continue
                distance = distanceSongs(centroids[k], songsDict[song]) #currently: song obj, song obj
                if distance < min_distance:
                    min_distance = distance
                    min_centroid = k
            assignments[min_centroid].append(songsDict[song]) #currently trackid
        new_centroids = [CONST_FILLER_SONG for j in range(len(centroids))]
        totals = [len(assignments[j]) for j in xrange(0, len(centroids))] #TODO THIS SECTION ONLY ASSUMES YEAR --------
    	for j in xrange(0, len(centroids)): #look at a centroid (cluster)
            if len(assignments[j]) == 0: continue #TODO THIS SHOULD POSSIBLY CHANGE?
            clusterTotalYears = 0.
            for k in xrange(0, totals[j]): #for each assignment in that cluster
                clusterTotalYears += assignments[j][k].year
            avgYear = clusterTotalYears / totals[j]
            newCentroid = Song('nan')
            newCentroid.year = avgYear #TODO SHOULD I DO INT? OR NA
            new_centroids[j] = newCentroid
        if new_centroids == centroids: #uh oh this might not work since im comparing objects with instances
            break
        centroids = new_centroids
    return centroids, assignments

#-------------------------------------------------------------------------------
#getTrackidList()

#Creates a list of all trackid's for future use.
#-------------------------------------------------------------------------------
def getTrackidList(inputPath):
    ret = []
    f = open(inputPath, 'r')
    for line in f.readlines():
        id = line[0:18]
        ret.append(id)
    return ret

#-------------------------------------------------------------------------------
#getTrackidFromTrackName(name)

#The idea of this function is given the name of a song, get its trackid.
#Assume that tracknames are perfectly formatted from playlist
#-------------------------------------------------------------------------------
def getTrackidFromTrackName(name):
    f = open('./AdditionalFiles/subset_unique_tracks.txt', 'r')
    for line in f.readlines():
        processed = re.split('<SEP>', line)
        if processed[3].strip().lower() == name.lower():
            return processed[1]
    return "Could not find that song."

#-------------------------------------------------------------------------------
#class Song

#Contains relevant information about a song, maybe represent with sparse vectors
#TODO So far just contains metadata, not audio data
#-------------------------------------------------------------------------------
class Song:
    def __init__(self, trackid):
        self.trackid = trackid
        self.album = "n/a"
        self.title = "n/a"
        self.artistid = 0
        self.artistName = "n/a"
        self.similarArtists = []
        self.duration = 0
        self.year = 0 #<=============currently being used
        # self.terms = []

        # self.maxLoudness = 0
        # self.maxLoudnessDuration = 0
        # self.segmentPitches = 0
        #Not sure what other fields to use/how to handle them

    #For debugging purposes
    def printSong(self):
        print "------------INSTANCE OF SONG-------------"
        print "TRACKID: "  + str(self.trackid)
        print "ALBUM: "  + str(self.album)
        print "TITLE: "  + str(self.title)
        print "ARTISTID: "  + str(self.artistid)
        print "ARTISTNAME: "  + str(self.artistName)
        print "SIMILAR_ARTISTS: "  + str(self.similarArtists)
        print "DURATION: "  + str(self.duration)
        print "YEAR: "  + str(self.year)
        # print "TERMS: "  + str(self.terms)
        print "-----------------------------------------"

    def populateFields(self): #Currently only contains the most basic metadata
        filename = './data/'+ self.trackid[2] + '/' + self.trackid[3] + '/' + self.trackid[4] + '/' + self.trackid + ".h5"
        f = h5py.File(filename, 'r') #introduce something to throw on failure
        metadata = f['metadata']
        songMetaData = metadata['songs'].value[0]
        self.album = songMetaData[14]
        self.title = songMetaData[18]
        self.artistid = songMetaData[5]
        self.artistName = songMetaData[9]
        self.similarArtists = np.asarray(metadata['similar_artists'].value[0])
        # self.duration =
        self.year = f['musicbrainz']['songs'].value[0][1]
        # self.terms = metadata['artist_terms'].value[0] #add weight and frequency potentially

        #TODO: MUSIC FEATURES

    def __eq__(self, other): #TODO: FOR NOW ONLY CHECKS YEAR EQUIVALENCE
        return self.year == other.year

    def __ne__(self,other):
        return self.year != other.year

#-------------------------------------------------------------------------------
#populateSongs(songList)

#Takes a songList which is an array containing trackid's and populates Song
#classes for all the songs in songList
#-------------------------------------------------------------------------------
def populateSongs(songList):
    songObjectArr = {}
    i= 0.
    for song in songList:
        i += 1
        obj = Song(song)
        obj.populateFields()
        songObjectArr.update({song:obj})
        sys.stdout.write("\r%d/10000 Songs Read" % i)
        sys.stdout.flush()
    return songObjectArr

#-------------------------------------------------------------------------------
#save

#Saves a python object songsDict under a pickle file with name name
#-------------------------------------------------------------------------------
def save(object, name):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

#-------------------------------------------------------------------------------
#load

#loads the pickle file with name name
#-------------------------------------------------------------------------------
def load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#-------------------------------------------------------------------------------
#readAndSavePickle

#reads a dataset of tracks
#-------------------------------------------------------------------------------
def readAndSavePickle(inputPath):
    print "---------------------Reading Data from 10,000 songs---------------------"
    trackidList = getTrackidList(inputPath)
    songsArray = populateSongs(trackidList)
    print "\n"
    print "Saving songs in a pickle file..."
    save(songsArray, "songsDict")
    print "\n"
    print "--------------------------Data Reading Complete--------------------------"
    print "Access Songs data with songsArray[trackid]. This will return a Song class"

#-------------------------------------------------------------------------------
#Live Scripts to actually do stuff:
#-------------------------------------------------------------------------------
# readAndSavePickle('./AdditionalFiles/subset_unique_tracks.txt')
CONST_FILLER_SONG = Song('filler')
CONST_FILLER_SONG.year = float('inf')

print "-----------------------Loading Song Data from Pickle------------------------"
newDict = load("songsDict")
print "--------------------------Data Loading is Complete--------------------------"
centroids, assignments = kMeansAllSongs(newDict, 5) #the centroids are not always the same for year
#each centroid is a Song object. Each element in assignments is an array of Song objects
#might need to change this so that trackid is readily accessible

for centroid in centroids: #To see what the years are for the centroids
    print centroid.year
