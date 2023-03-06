import json
import logging
import numpy as np
import os
import pandas as pd
from urllib import request

import editdistance as editdistance
import re

def remove_urls(loc_name):
    #BUG: actually remove urls
    return loc_name


class LocationFinder:
    key = os.getenv('MAPS_API_KEY')
    pre = 'https://maps.googleapis.com/maps/api/geocode/json?address='
    post = '&key='+key

    @staticmethod
    def find_country(location_name):
        logging.debug(f'{location_name = }')
        location_name = location_name.lower()
        location_name = location_name.replace('land of the free','')
        location_name = location_name.replace('the','')
        location_name = location_name.replace('worldwide','')
        location_name = re.sub('n\.?y\.?c\.?', 'new york city', location_name)
        location_name = re.sub('(united states of )?america', 'united states', location_name)
        location_name = re.sub('u\.?s\.?a\.?', 'united states', location_name)
        location_name = location_name.replace('united kingdom', 'uk')
        location_name = location_name.replace('republic of texas', 'texas')
        location_name = location_name.replace('european union', '')
        location_name = remove_urls(location_name)
        url_location = '+'.join(re.compile('[\w/]+').findall(location_name)) # removes emojis
        logging.debug(f'{url_location = }')

        location = dict()
        location['address'] = None
        location['country'] = None
        if url_location.replace(' ', '') != '':
            logging.debug(url_location)
            reply = request.urlopen(LocationFinder.pre+url_location+LocationFinder.post).read().decode("utf-8")

            json_reply = json.loads(reply)
            logging.debug(f'{json_reply = }')
            if json_reply['status'] == 'OK':
                best_res = None
                min_edit_distance = 1000
                for res in json_reply['results']:
                    address = res['formatted_address'].lower()
                    dist = editdistance.eval(location_name, address)
                    if dist < len(address)*(3/4) or location_name in address or address in location_name:
                        if 'locality' or 'country' in res['types']:
                            if(dist < min_edit_distance):
                                min_edit_distance = dist
                                best_res = res

                if best_res is not None:
                    country = None
                    for comp in best_res['address_components']:
                        if 'country' in comp['types']:
                            country = comp['long_name']
                    if 'locality' in best_res['types']:
                        location['address'] = best_res['formatted_address']
                    location['country'] = country
                    logging.info("Location found for '" + location_name + "': " + str(location['address']) + ", " + str(location['country']))
                    return country
                else:
                    logging.info("No location found for '" + location_name + "'")
        return np.nan


    @staticmethod
    def insert_countries(df, country_col_name):
        no_country = df[df[country_col_name].isna()]
        no_location = no_country['location'].isna()
        no_country_yes_location = no_country[~no_location]
        df[country_col_name] = no_country_yes_location['location'].apply(LocationFinder.find_country)

        return df

def main():
    logging.basicConfig(filename='gmaps.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s - %(message)s')

    READ_CSV = 'dev_users.csv'
    print(f'Reading: {READ_CSV}')
    df = pd.read_csv(READ_CSV)

    df['country'] = np.nan
    loc_col = 'country'

    df = LocationFinder.insert_countries(df, loc_col)

    df.to_csv('locations.csv', index=False)

if __name__ == "__main__":
    main()

