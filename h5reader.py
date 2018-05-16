#-------------------------------------------------------------------------------
#h5reader.py

#IMPORTANT: PLACE THIS FILE IN THE ORIGIN FOLDER
#folder that contains ADDITIONALFILES and DATA

#This file recursively opens all h5 files in the data folder of the dataset and
#populates Song classes for each of the 10,000 songs. Each song class contains
#relevant feature information for clustering and training a Bayesian classifier

#Approximate runtime of a single file open: 0.33 seconds TODO: speed this up
#TODO: idea, load in the background, in the foreground, only load as needed
#-------------------------------------------------------------------------------
import h5py

#-------------------------------------------------------------------------------
#getTrackidList()

#Creates a list of all trackid's for future use.
#-------------------------------------------------------------------------------
def getTrackidList():
    ret = []
    f = open('./AdditionalFiles/subset_unique_tracks.txt', 'r')
    # maxTracks = 10000 #FOR TESTING (4 seconds for 100, 11 seconds for 1,000, 1 min 56 sec for 10,000)
    # i = 0 #FOR TESTING
    for line in f:
        id = line[0:line.find('<SEP>')]
        ret.append(id)
        # i += 1 #FOR TESTING
        # if i == maxTracks: break #FOR TESTING
    return ret

#-------------------------------------------------------------------------------
#getTrackidFromTrackName(name)

#The idea of this function is given the name of a song, get its trackid.
#-------------------------------------------------------------------------------
def getTrackidFromTrackName(name):
    pass

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
        self.songid = "n/a"
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
        print "SONGID: "  + str(self.songid)
        print "SIMILAR_ARTISTS: "  + str(self.similarArtists)
        print "DURATION: "  + str(self.duration)
        print "YEAR: "  + str(self.year)
        # print "TERMS: "  + str(self.terms)
        print "-----------------------------------------"

    def populateFields(self):
        filename = './data/'+ self.trackid[2] + '/' + self.trackid[3] + '/' + self.trackid[4] + '/' + self.trackid + ".h5"
        f = h5py.File(filename, 'r')
        metadata = f['metadata']
        songMetaData = metadata['songs']
        self.album = songMetaData.value[0][14]
        self.title = songMetaData.value[0][18]
        self.artistid = songMetaData.value[0][5]
        self.artistName = songMetaData.value[0][9]
        self.songid = songMetaData.value[0][17]
        self.similarArtists = metadata['similar_artists'].value[0]
        # self.duration =
        self.year = f['musicbrainz']['songs'].value[0][1]
        # self.terms = metadata['artist_terms'].value[0] #add weight and frequency potentially

#-------------------------------------------------------------------------------
#populateSongs(songList)

#Takes a songList which is an array containing trackid's and populates Song
#classes for all the songs in songList
#-------------------------------------------------------------------------------
def populateSongs(songList):
    songObjectArr = []
    for song in songList:
        obj = Song(song)
        obj.populateFields()
        songObjectArr.append(obj)
    return songObjectArr

#-------------------------------------------------------------------------------
#Live Scripts to actually do stuff:
#-------------------------------------------------------------------------------
trackidList = getTrackidList()
songsArray = populateSongs(trackidList)
