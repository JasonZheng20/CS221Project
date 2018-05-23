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

#gets the lowest distance song to a centroid (fun function)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#randomCentroids(songsDict, numCentroids)

#Takes songsDict, and constructs numCentroids # of random centroids
#-------------------------------------------------------------------------------
def randomCentroids(songsDict, numCentroids):
    centroids = []
    for i in xrange(numCentroids):
        centroids.append(songsDict[random.choice(songsDict.keys())]) #currently song obj
    return centroids

#-------------------------------------------------------------------------------
#constructCentroid()

#Takes a dictionary indicating the average of all of the fields of all songs
#in a cluster, and generates a new centroid Song containing that information
#-------------------------------------------------------------------------------
def constructCentroid(avgFieldsDict):
    newCentroid = Song('centroid')
    for field in avgFieldsDict:
        newCentroid[field] = avgFieldsDict[field]
    return newCentroid

#-------------------------------------------------------------------------------
#distanceSongs(song1, song2)

#Takes two song objects and computes their distance
#Currently computes distance solely by year
#TODO: Create an algorithm to be able to weight all of the terms properly to get accurate distances
#TODO: We don't really have labeled data for song distance, so it will have to be unsupervised(?)
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
            if len(assignments[j]) == 0: continue
            #------------------------------------------------------------------------------------
            #TODO: Generalize the averaging to multiple fields, check for uninitialized values (see Song class for details)
            #------------------------------------------------------------------------------------
            clusterTotalYears = 0.
            for k in xrange(0, totals[j]): #for each assignment in that cluster
                clusterTotalYears += assignments[j][k].year
            avgYear = clusterTotalYears / totals[j]
            #I have implemented a constructCentroid function here that takes in a dict of values
            #------------------------------------------------------------------------------------
            newCentroid = Song('nan')
            newCentroid.year = avgYear
            #------------------------------------------------------------------------------------
            new_centroids[j] = newCentroid
        if new_centroids == centroids:
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
#-------------------------------------------------------------------------------
class Song:
    def __init__(self, trackid):
        self.trackid = trackid
        self.album = "n/a"
        self.title = "n/a"
        self.artistid = "nan" #If not provided, it is 'nan'
        self.artistName = "n/a"
        self.similarArtists = []
        self.terms = []

        #Numerical Fields --TODO: CHECK WHAT DEFAULTS ARE
        self.duration = 0 #assume always provided
        self.year = 0 #If it is not provided, it is 0
        self.key = 0 #Same deal as mode, see below, though key can take on more values
        self.keyConfidence = 0 #if 0.0 do not use self.key
        self.generalLoudness = 0 #assume it is always given
        self.mode = 0 #appears to always be 0 or 1, don't include if modeConfidence is 0.0
        self.modeConfidence = 0 #if modeConfidence is 0.0, assume don't use mode info, even if provided TODO: experiment with using despite 0.0
        self.tempo = 0 #if not provided, it will be 0
        self.timeSigniature = 0 #if not provided it will be 0
        self.timeSigniatureConfidence = 0 #if this is 0.0 do not use time sig

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
        print "TERMS: "  + str(self.terms)
        print "KEY: " + str(self.key)
        print "KEY_CONFIDENCE: " + str(self.keyConfidence)
        print "LOUDNESS: " + str(self.generalLoudness)
        print "MODE: " + str(self.mode)
        print "MODE_CONFIDENCE: " + str(self.modeConfidence)
        print "TEMPO: " + str(self.tempo)
        print "TIME_SIG: " + str(self.timeSigniature)
        print "TIME_SIG_CONFIDENCE: " + str(self.timeSigniatureConfidence)
        print "-----------------------------------------"

    def populateFields(self):
        filename = './data/'+ self.trackid[2] + '/' + self.trackid[3] + '/' + self.trackid[4] + '/' + self.trackid + ".h5"
        f = h5py.File(filename, 'r')
        metadata = f['metadata']
        songMetaData = metadata['songs'].value[0]
        songsMeta = f['analysis']['songs'].value[0]
        self.album = songMetaData[14]
        self.title = songMetaData[18]
        self.artistid = songMetaData[5]
        self.artistName = songMetaData[9]
        self.similarArtists = np.asarray(metadata['similar_artists'].value[0])
        self.duration = songsMeta[3]
        self.year = f['musicbrainz']['songs'].value[0][1]
        self.key = songsMeta[21]
        self.keyConfidence = songsMeta[22] #WHAT TO DO HERE
        self.generalLoudness = songsMeta[23]
        self.mode = songsMeta[24]
        self.modeConfidence = songsMeta[25] #WHAT TO DO HERE
        self.tempo = songsMeta[27]
        self.timeSigniature = songsMeta[28]
        self.timeSigniatureConfidence = songsMeta[29] #WHAT TO DO HERE
        self.terms = metadata['artist_terms'].value #add weight and frequency potentially
        #TODO: weight for terms
        #TODO: frequency for terms

    def __eq__(self, other): #TODO: FOR NOW ONLY CHECKS YEAR EQUIVALENCE
#------------------------------------------------------------------------------------
#TODO: Update this so equivalence requires all fields are equal
#------------------------------------------------------------------------------------
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
        # obj.printSong() #Take a look at this if you would like to further scrutinize extreme values
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
# readAndSavePickle('./AdditionalFiles/subset_unique_tracks.txt') #(~1 min 50 seconds)
CONST_FILLER_SONG = Song('filler')
CONST_FILLER_SONG.year = float('inf')

print "-----------------------Loading Song Data from Pickle------------------------"
newDict = load("songsDict")
print "--------------------------Data Loading is Complete--------------------------"
centroids, assignments = kMeansAllSongs(newDict, 5) #the centroids are not always the same for year
# #each centroid is a Song object. Each element in assignments is an array of Song objects
# #might need to change this so that trackid is readily accessible
#
# for centroid in centroids: #To see what the years are for the centroids
#     print centroid.year
