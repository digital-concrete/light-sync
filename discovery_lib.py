# coding: utf-8
"""
Helper for connecting to the bridge. If we don't
have a valid username for the bridge (ip) we are trying
to use, this will cause one to be generated.
"""

from __future__ import print_function
import sys
import json
import requests


# NUPNP adress
HUE_NUPNP = "https://discovery.meethue.com"

class IPError(Exception):
    ''' Raise when the Hue Bridge IP address cannot be resolved '''


class DiscoveryLib:
    def __init__(self):
         "init"

   ################
    # HTTP METHODS #
    ################

    # GET Request
    def get(self, url):
        response = requests.get(url, timeout = 10)
        return self.responseData(response)

    # PUT Request
    def put(self, url, payload):
        response = requests.put(url, data = json.dumps(payload))
        return self.responseData(response)

    # POST Request
    def post(self, url, payload):
        response = requests.post(url, data = json.dumps(payload))
        return self.responseData(response)


    #############
    # HUE SETUP #
    #############

    # Gets bridge IP using Hue's NUPNP site. Device must be on the same network as the bridge
    def getBridgeIP(self):
        try:
            return self.get(HUE_NUPNP)['json'][0]['internalipaddress']
        except:
            raise IPError('Could not resolve Hue Bridge IP address. Please ensure your bridge is connected')

    # If given a brige IP as a constructor parameter, this validates it
    def validateIP(self, ip):
        try:
            data = self.get('http://{}/api/'.format(ip))
            if not data['ok']:
                raise IPError('Invalid Hue Bridge IP address')
        except (requests.exceptions.ConnectionError,
                requests.exceptions.MissingSchema,
                requests.exceptions.ConnectTimeout):
            raise IPError('Invalid Hue Bridge IP address')

        return ip


    # Takes HTTP request response and returns pertinent information in a dict
    def responseData(self, response):
        data = {'status_code': response.status_code, 'ok': response.ok}
        if response.ok:
            data['json'] = response.json()
        return data

    #####################
    # CUSTOM EXCEPTIONS #
    #####################

