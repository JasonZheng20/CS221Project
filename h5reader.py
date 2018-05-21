#-------------------------------------------------------------------------------
#h5reader.py

#IMPORTANT: PLACE THIS FILE IN THE ORIGIN FOLDER
#folder that contains ADDITIONALFILES and DATA

#This file recursively opens all h5 files in the data folder of the dataset and
#populates Song classes for each of the 10,000 songs. Each song class contains
#relevant feature information for clustering and training a Bayesian classifier

#TODO LIST:
#Write an initial all song clustering algorithm (one time) Keep track of whats in each cluster
#Once we have centroids, only search closest cluster to song to find similar songs

# I need cluster objects so james can update
#Cluster of songs, centroids, and songs corresponding to centroid, (trackids in cluster)

#When caching, write to file the centroid, and all trackids to centroid

#-------------------------------------------------------------------------------
import h5py
import numpy as np
import time
import sys
import re
import pickle

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
        # self.songid = "n/a"
        self.similarArtists = []
        self.duration = 0
        self.year = 0
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
        # print "SONGID: "  + str(self.songid)
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
        # self.songid = songMetaData.value[0][17]
        self.similarArtists = np.asarray(metadata['similar_artists'].value[0])
        # self.duration =
        self.year = f['musicbrainz']['songs'].value[0][1]
        # self.terms = metadata['artist_terms'].value[0] #add weight and frequency potentially

        #TODO: MUSIC FEATURES

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
print "--------------------------Loading Data from Pickle--------------------------"
newArray = load("songsDict")
print "--------------------------Data Loading is Complete--------------------------"

# print newArray['TRBGCWM12903CF5BF7'].title #FOR THE TEST
