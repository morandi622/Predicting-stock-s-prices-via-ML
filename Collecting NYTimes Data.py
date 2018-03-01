
from newsapi.articles import Articles
from newsapi.sources import Sources
key = '96af62a035db45bda517a9ca62a25ac3'
a,s = Articles(API_KEY=key),Sources(API_KEY=key)
s.all() # get all sources offered by newsapi

a.get(source='the-new-york-times')
s.get(category='technology', language='en', country='US')




from newsapi import NewsAPI

key = '96af62a035db45bda517a9ca62a25ac3'
params = {}
api = NewsAPI(key)
sources = api.sources(params)
articles = api.articles(sources[0]['id'], params)

################ NY Times API #############################################


import sys, csv, json
reload(sys)
sys.setdefaultencoding('utf8')




















"""
About:
Python wrapper for the New York Times Archive API 
https://developer.nytimes.com/article_search_v2.json
"""

class APIKeyException(Exception):
    def __init__(self, message): self.message = message 

class InvalidQueryException(Exception):
    def __init__(self, message): self.message = message 

class ArchiveAPI(object):
    def __init__(self, key=None):
        """
        Initializes the ArchiveAPI class. Raises an exception if no API key is given.
        :param key: New York Times API Key
        """
        self.key = key
        self.root = 'http://api.nytimes.com/svc/archive/v1/{}/{}.json?api-key={}' 
        if not self.key:
            nyt_dev_page = 'http://developer.nytimes.com/docs/reference/keys'
            exception_str = 'Warning: API Key required. Please visit {}'
            raise NoAPIKeyException(exception_str.format(nyt_dev_page))

    def query(self, year=None, month=None, key=None,):
        """
        Calls the archive API and returns the results as a dictionary.
        :param key: Defaults to the API key used to initialize the ArchiveAPI class.
        """
        if not key: key = self.key
        if (year < 1882) or not (0 < month < 13):
            # currently the Archive API only supports year >= 1882
            exception_str = 'Invalid query: See http://developer.nytimes.com/archive_api.json'
            raise InvalidQueryException(exception_str)
        url = self.root.format(year, month, key)
        r = requests.get(url)
        return r.json()


api = ArchiveAPI('0ba6dc04a8cb44e0a890c00df88c393a')
import requests, json,time


for year in range(2007,2018+1):
    for month in range(1,12+1):
        nynews = api.query(year, month)
        file = 'data/nytimes/' + str(year) + '-' + '{:02}'.format(month) + '.json'
        with open(file, 'w') as out:
            json.dump(nynews, out)
        out.close()
        time.sleep(60)
        

    
