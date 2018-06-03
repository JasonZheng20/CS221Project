#-------------------------------------------------------------------------------
#h5reader.py

#This file contains code to load and save Song objects from song datasets
#IDEA: http://kldavenport.com/the-cost-function-of-k-means/
#IDEA: WORD MAP
# Using a different distance function other than (squared) Euclidean distance may stop the algorithm from converging.[citation needed] Various modifications of k-means such as spherical k-means and k-medoids have been proposed to allow using other distance measures.
#-------------------------------------------------------------------------------
import h5py
import time
import sys
import getopt
import re
import pickle
import random
import math
import numpy as np
from scipy import spatial

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
def constructCentroid(d, k, l, m, t, s,terms):
    newCentroid = Song('centroid')
    newCentroid.duration = d
    newCentroid.key = k
    newCentroid.generalLoudness = l
    newCentroid.mode = m
    newCentroid.tempo = t
    newCentroid.timeSigniature = s
    newCentroid.terms = terms
    return newCentroid

#-------------------------------------------------------------------------------
#distanceSongs(song1, song2)

#Takes two song objects and computes their distance
#-------------------------------------------------------------------------------
def distanceSongs(song1, song2, weights):
    distance = 0
    distance += weights[0]*abs(song1.duration - song2.duration)
    if song2.keyConfidence > 0:
        distance += weights[1]*abs(song1.key - song2.key)
    distance += weights[2]*abs(song1.generalLoudness - song2.generalLoudness)
    if song2.modeConfidence > 0:
        distance += weights[3]*abs(song1.mode - song2.mode)
    if song2.tempo != 0:
        distance += weights[4]*abs(song1.tempo - song2.tempo)
    if song2.timeSigniatureConfidence > 0:
        distance += weights[5]*abs(song1.timeSigniature - song2.timeSigniature)
    distance += weights[6]*term_distance(song1,song2)
    return distance


def term_distance(centroid, song):
    song_term_keys = set(song.terms.keys())
    distance = 0.1
    total = 0.1
    for term in song_term_keys:
        cent_val = centroid.terms.get(term, (0,0))
        song_val = song.terms[term]
        distance += math.sqrt((cent_val[0]-song_val[0])**2 + (cent_val[1]-song_val[1])**2)
        total += 1
    return distance/total

#-------------------------------------------------------------------------------
#kMeansAllSongs(songsDict, numCentroids)

#Takes a dictionary of trackid:Song object, and creates a clustering of similar
#songs using K-means clustering algorithm.
#currently storing each thing as the song object itself, due to computation ease
#returns centroids and clusterings

#TODO: update kMeans so that it works on an arbitrary number of fields, as described above
#-------------------------------------------------------------------------------
def kMeansAllSongs(songsDict, weights, numCentroids = 5, T = 1000):
    centroids = randomCentroids(songsDict, numCentroids)
    for i in xrange(0, T):
        assignments = [[] for j in range(len(centroids))]
        for song in songsDict:
            min_distance = float('inf')
            min_centroid = 0
            for k in xrange(0, len(centroids)):
                if centroids[k] == CONST_FILLER_SONG: continue
                distance = distanceSongs(centroids[k], songsDict[song], weights)
                if distance < min_distance:
                    min_distance = distance
                    min_centroid = k
            assignments[min_centroid].append(songsDict[song]) #currently trackid
        new_centroids = [CONST_FILLER_SONG for j in range(len(centroids))]
        totals = [len(assignments[j]) for j in xrange(0, len(centroids))]
        for j in xrange(0, len(centroids)): #look at a centroid (cluster)
            thisCluster = assignments[j]
            if len(thisCluster) == 0: continue
            clusterTotalDuration = 0.
            clusterTotalKeys = 0.
            clusterTotalLoudness = 0.
            clusterTotalMode = 0.
            clusterTotalTempo = 0.
            clusterTotalTimeSig = 0.
            numFieldsGiven = [1,1,1,1,1]

            total_terms = 0.1
            new_terms = {}
            for k in xrange(0, totals[j]): #for each assignment in that cluster
                thisSong = thisCluster[k]
                clusterTotalDuration += thisSong.duration
                if thisSong.keyConfidence > 0:
                    clusterTotalKeys += thisSong.key
                    numFieldsGiven[1] += 1
                clusterTotalLoudness += thisSong.generalLoudness
                if thisSong.modeConfidence > 0:
                    clusterTotalMode += thisSong.mode
                    numFieldsGiven[2] += 1
                if thisSong.tempo != 0:
                    clusterTotalTempo += thisSong.tempo
                    numFieldsGiven[3] += 1
                if thisSong.timeSigniatureConfidence != 0:
                    clusterTotalTimeSig += thisSong.timeSigniature
                    numFieldsGiven[4] += 1
                for term in thisSong.terms.keys():
                    song_val = thisSong.terms[term]
                    values = new_terms.get(term,(0,0))
                    new_terms[term] = (song_val[0]+values[0], song_val[0]+values[1])
                total_terms+=1
            avgDuration = clusterTotalDuration / numFieldsGiven[1]
            avgKey = clusterTotalKeys / totals[j]
            avgLoudness = clusterTotalLoudness / totals[j]
            avgMode = clusterTotalMode / numFieldsGiven[2]
            avgTempo = clusterTotalTempo / numFieldsGiven[3]
            avgTimeSig = clusterTotalTimeSig / numFieldsGiven[4]




            for term in new_terms.keys():
                values = new_terms[term]
                new_terms[term] = ((values[0]+.1)/total_terms,(values[1]+.1)/total_terms)

            new_centroids[j] = constructCentroid(avgDuration, avgKey, avgLoudness, avgMode, avgTempo, avgTimeSig, new_terms)
        if new_centroids == centroids:
            print "CONVERGENCE AFTER " + str(i) + " TRIALS"
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
    f = open('../data/MillionSongSubset/AdditionalFiles/subset_unique_tracks.txt', 'r')
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
        self.terms = {}

        #Numerical Fields
        self.duration = 0 #assume always provided
        self.key = 0 #Same deal as mode, see below, though key can take on more values
        self.keyConfidence = 0 #if 0.0 do not use self.key
        self.generalLoudness = 0 #assume it is always given
        self.mode = 0 #appears to always be 0 or 1, don't include if modeConfidence is 0.0
        self.modeConfidence = 0 #if modeConfidence is 0.0, assume don't use mode info
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
        # print "YEAR: "  + str(self.year)
        print "KEY: " + str(self.key)
        print "LOUDNESS: " + str(self.generalLoudness)
        print "MODE: " + str(self.mode)
        print "TEMPO: " + str(self.tempo)
        print "TIME_SIG: " + str(self.timeSigniature)
        print "TERMS: " + str(self.terms)
        print "-----------------------------------------"

    def concisePrint(self):
        # print [self.year, self.duration, self.key, self.generalLoudness, self.mode, self.tempo, self.timeSigniature]
        print [self.duration, self.key, self.generalLoudness, self.mode, self.tempo, self.timeSigniature]

    def populateFields(self):
        filename = '../data/MillionSongSubset/data/'+ self.trackid[2] + '/' + self.trackid[3] + '/' + self.trackid[4] + '/' + self.trackid + ".h5"
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
        self.key = songsMeta[21]
        self.keyConfidence = songsMeta[22] #WHAT TO DO HERE
        self.generalLoudness = songsMeta[23]
        self.mode = songsMeta[24]
        self.modeConfidence = songsMeta[25] #WHAT TO DO HERE
        self.tempo = songsMeta[27]
        self.timeSigniature = songsMeta[28]
        self.timeSigniatureConfidence = songsMeta[29] #WHAT TO DO HERE
        temp_terms = metadata['artist_terms'].value
        temp_term_weight = metadata['artist_terms_weight'].value
        temp_term_freq = metadata['artist_terms_freq'].value
        for i in range(len(metadata['artist_terms'].value)):
            self.terms[temp_terms[i]]=(temp_term_weight[i],temp_term_freq[i])

    def __eq__(self, other):
        return self.duration == other.duration and self.key == other.key and self.generalLoudness == other.generalLoudness and self.mode == other.mode and self.tempo == other.tempo and self.timeSigniature == other.timeSigniature

    def __ne__(self,other):
        return self.duration == other.duration or self.key == other.key or self.generalLoudness == other.generalLoudness or self.mode == other.mode or self.tempo == other.tempo or self.timeSigniature == other.timeSigniature

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
    save(songsArray, "../songsDict")
    print "\n"
    print "--------------------------Data Reading Complete--------------------------"
    print "Access Songs data with songsArray[trackid]. This will return a Song class"

#-------------------------------------------------------------------------------
#succinctLoss()

#Does the same as clusteringLoss without running k-means
#-------------------------------------------------------------------------------
def succinctLoss(assignments, numClusters):
    unevenness = 0.
    assignmentLengths = sorted([len(cluster) for cluster in assignments])
    for j in xrange(0, len(assignments)):
        unevenness += ((10000/numClusters)-assignmentLengths[j])**2
    return unevenness

#-------------------------------------------------------------------------------
#readAndSaveClusters

#runs K-means and saves a pickle file of the clusterings
#-------------------------------------------------------------------------------
def readAndSaveClusters(songsDict, numClusters = 5):
    print "----------------------------Running K-means------------------------------"
    weightsFromLearn = [-0.060030265506692514, -0.44383045710403024, 3.4490915718474104, 3.695904771678345, 4.6554185014126634, 5.0456856919515936, 3.139756095232522]
    centroids, assignments = kMeansAllSongs(songsDict, weightsFromLearn, numClusters, 75)
    loss = succinctLoss(assignments, numClusters)
    print "Clustered with Loss: " + str(loss)
    print "---------------------------Completed K-means-----------------------------"
    print "----------------------Saving Clusterings to Pickle-----------------------"
    save(centroids, "../centroids")
    save(assignments, "../assignments")
    print "---------------------Done Saving Clusters to Pickle----------------------"
    return centroids, assignments

#-------------------------------------------------------------------------------
#clusteringLoss()

# We define clustering loss as x,
# between each trial, we have a deviation x, which
# is defined as the sum of all squared differences between #items in 5 clusters
# divided by the number of trials.
#-------------------------------------------------------------------------------
def calculateClusteringDeviation(songsDict, weights):
    numClusters = 5
    unevenness = 0.
    centroids, assignments = kMeansAllSongs(songsDict, weights, numClusters, 75)
    previous_assignmentLengths = sorted([len(cluster) for cluster in assignments])
    for j in xrange(0, len(assignments)):
        unevenness += ((10000/numClusters)-previous_assignmentLengths[j])**2
    return unevenness

#DEPRECATED: Calculate consistency:
    # print previous_assignmentLengths #comment out when ready
    # for i in xrange(0,5):
    #     centroids, assignments = kMeansAllSongs(songsDict, weights, 5, 75)
    #     assignmentLengths = sorted([len(cluster) for cluster in assignments])
    #     print assignmentLengths #comment out when ready
    #     for j in xrange(0,len(assignmentLengths)):
    #         deviation += (assignmentLengths[j] - previous_assignmentLengths[j])**2
    #         unevenness += ((10000/numClusters)-assignmentLengths[j])**2
    #     previous_assignmentLengths = assignmentLengths
    # print unevenness #comment out when ready
#-------------------------------------------------------------------------------
#learnDistanceWeights()

#Function to learn distance weights for K-means computation to minimize variance
#of clustering across K-means computations. This is to ensure that we have most
#accurate clusterings possible.
#-------------------------------------------------------------------------------
def learnDistanceWeights(songsDict):
    weights = [-0.060030265506692514, -0.44383045710403024, 3.4490915718474104, 3.695904771678345, 4.6554185014126634, 5.0456856919515936, 3.139756095232522]
    prev_distance = calculateClusteringDeviation(songsDict,weights)
    print("previous best weights: "+str(weights))
    print("distance: " + str(prev_distance))
    iterations = 25
    step_size = .05
    weight_change = [step_size*iterations for j in range(len(weights))]
    best_weights = []
    best_distance = float('inf')
    for i in range(iterations):
        for j in range(len(weights)):
            weights[j]+=weight_change[j]
            distance = calculateClusteringDeviation(songsDict,weights)
            value = -1 if (weight_change[j]<0) else 1
            if(distance - prev_distance >0):
                value = -1 * value
            weight_change[j] = value*(step_size*iterations/float(i+1.0))
            prev_distance = distance
            if(distance<best_distance):
                best_weights = weights
                best_distance = distance
            if(distance > 40000000 and best_distance < 40000000):
                weights = best_weights
        print prev_distance
        print weights
        print("\n")
    print("It's the final weight\n")
    print("time for the best weights\n")
    print best_weights

#-------------------------------------------------------------------------------
#Main Function:

# Usage:
# -s to load data and save songs
# -c to load songs and save clusters
# -l to load songs and learn weights
# -i to load saved cluster data
#-------------------------------------------------------------------------------
CONST_FILLER_SONG = Song('filler')
CONST_FILLER_SONG.year = float('inf')

def main():
    (options, args) = getopt.getopt(sys.argv[1:], 'scli')
    if ('-s','') in options: #save the Songs Pickle
        readAndSavePickle('../data/MillionSongSubset/AdditionalFiles/subset_unique_tracks.txt') #(~1 min 50 seconds)
    elif ('-c', '') in options: #save the Clusters Pickle
        print "-----------------------Loading Song Data from Pickle------------------------"
        newDict = load("../songsDict")
        print "--------------------------Data Loading is Complete--------------------------"
        centroids, assignments = readAndSaveClusters(newDict)
    elif ('-l', '') in options:
        print "-----------------------Loading Song Data from Pickle------------------------"
        newDict = load("../songsDict")
        print "--------------------------Data Loading is Complete--------------------------"
        print "------------------------Learning Parameter Weights--------------------------"
        learnDistanceWeights(newDict)
    elif ('-i', '') in options:
        print "---------------------Loading Clusterings from Pickle------------------------"
        centroids = load("../centroids")
        assignments = load("../assignments")
        print "----------------------Loaded Clusterings from Pickle------------------------"
    else:
        print "Usage: -s to read and save songs, -c to load songs and save clusters, -l to load songs and learn weights, -i to load saved cluster data"


if __name__ == "__main__":
    main()
