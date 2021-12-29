#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pandas as pd
import numpy as np


original = "./final_project/final_project_dataset.pkl"
destination = "word_data_unix.pkl"


def unix_version(path_origin, path_dest):
    content = ''
    outsize = 0
    with open(path_origin, 'rb') as infile:
        content = infile.read()
        with open(path_dest, 'wb') as output:
            for line in content.splitlines():
                outsize += len(line) + 1
                output.write(line + str.encode('\n'))
    return path_dest

pickled_file = unix_version(original, destination)

infile = open(pickled_file, "rb")
enron_data = pickle.load(infile)

df = pd.DataFrame(enron_data).T
print(df.shape)
print(df.columns)
#print(np.sum(df['poi'] == 1))
#print(df.total_stock_value.head(40))
#print(df.loc['Prentice James'.upper()])
#print(df.loc['COLWELL WESLEY', 'from_this_person_to_poi'])
#print(df.loc['Skilling Jeffrey k'.upper(), ['exercised_stock_options']])
#for i in df.index.sort_values():
#    print(i)

#print(df.loc[['Lay Kenneth L'.upper(), 'Fastow Andrew s'.upper(), 'Skilling Jeffrey k'.upper()], 'total_payments'])
print(df['salary'])
#print(df['email_address'].isna().sum())
#print(df[df['salary'] == 'NaN'])
df['salary'] = df['salary'].replace('NaN', np.nan)
df['email_address'] = df['email_address'].replace('NaN', np.nan)
df['total_payments'] = df['total_payments'].replace('NaN', np.nan)

print(df.shape[0] - df['salary'].isna().sum())
print(df.shape[0] - df['email_address'].isna().sum())
print(df['total_payments'].isna().sum())
print(df['total_payments'].isna().sum()/df.shape[0])


pois = df[df['poi'] == True]
print(pois)
print(pois['total_payments'].isna().sum())