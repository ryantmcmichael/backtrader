# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 09:55:15 2022

@author: rtm
"""

api = '34495af4f36851a42dd7649d1e4dc2fb'
sym = 'AAPL'

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

import certifi
import json

def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

url = ("https://financialmodelingprep.com/api/v3/historical-chart/1hour/" +
       sym + "?apikey=" + api)
data = get_jsonparsed_data(url)