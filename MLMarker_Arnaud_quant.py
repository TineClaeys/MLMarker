import logging
import glob
import pandas as pd
import numpy as np
import os
import re
import warnings
# Filter out the specific warning message
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.")
from MLMarker_app import MLMarker
import pymongo
import sys


logging.FileHandler('MLMarker.log') #Create a log file
logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler('MLMarker.log')])


def extract_pxd(file_path):
    match = re.search(r'PXD.*?/(?<!zip/)', file_path) # Pattern that starts with PXD and ends with / but not with zip/
    if match:
        return match.group(0)[:-1]
    else:
        return None


def extract_filename(file_path):
    match = re.search(r'/(?!PXD).*?/', file_path) # Pattern that starts with / and everything else other than PXD and ends with /
    if match:
        return match.group(0)[1:-1]
    else:
        return None


    
def ionbot_first_to_prot_list(df):
        """Only search in the rows : q-value <= 0.01, database = T, proteins != CONTAMINANT and '|'  
        Return a list of uniprot id"""
        df = df[['proteins', 'database', 'q-value']]
        df = df[(df['database']=='T') & (df['q-value']<=0.01)]
        df = df[~df['proteins'].apply(lambda x: '|' in x or 'CONTAMINANT' in x)]
        uniprot_df = df['proteins'].apply(lambda uniprot: re.search(r'\)\)\(\((.*?)\)\)', uniprot).group(1) 
                                          if re.search(r'\)\)\(\((.*?)\)\)', uniprot) else None) # Found the uniprot id with a ))((.*)) pattern
        uniprot_df = uniprot_df.dropna() #Maybe not necessary 
        list_uniprot = (list(set(uniprot_df)))
        return list_uniprot


def csv_path(path_file):   
    """Return a dictionnary with the path of the csv files"""
    for result_dir in glob.iglob(path_file, recursive=True): #Recursive search of the zip files in the folder
        pxd = os.path.basename(os.path.split(result_dir)[0])
        for run_dir in os.listdir(result_dir):
            if run_dir.endswith('.mgf.gzip'): #Check if the file is a mgf.gz file
                result_file = os.path.join(result_dir, run_dir, "ionbot.first.csv")
                if os.path.exists(result_file):
                    yield pxd, run_dir, result_file


def MLMarker_Arnaud_quant(prot_list):
    data = np.ones((1, len(prot_list)))
    predict_df = pd.DataFrame(data, columns=prot_list)
    test = MLMarker(predict_df.iloc[0:1,:], binary = True)
    prediction = MLMarker.adjusted_shap_values_df(test, penalty_factor=0.5, n_preds=100)
    prediction = prediction.sum(axis=1).sort_values(ascending=False)
    df_predi = pd.DataFrame(prediction).transpose()
    return df_predi


def MLMarker_to_Dataframe(dico, lengths):
    """Return a dataframe with the data of the mgf.gz files after the MLMarker analysis"""

    df_big = pd.DataFrame()
    processed = set()
    for pxd, mgf_file, result_file in dico:
        if (pxd, mgf_file) in processed:
            logging.error(f'Duplicate in {pxd}/{mgf_file}')
            continue
        processed.add((pxd, mgf_file))

        try:
            with open(result_file) as f: 
                df_first = pd.read_csv(f)
                prot_list = ionbot_first_to_prot_quant(df_first, lengths)

            df_temp = MLMarker_Arnaud_quant(prot_list)
            df_temp['PXD_folder'] = pxd
            df_temp['filename'] = mgf_file

            df_big = pd.concat([df_big, df_temp])

        except Exception as e: #Not sure if this is the best way to handle errors
            logging.exception(f'Error in {pxd}/{mgf_file}')

    return df_big


def get_human_projects(mongo_uri):
    db = pymongo.MongoClient(mongo_uri).pipeline
    human_projects = {obj["accession"] for obj in db.metadata.find({"species": {"$in": ["Homo sapiens (Human)", "Homo sapiens (human)"]}})}
    modified_projects = {obj["accession"] for obj in db.file_registry.find({"modifications.ionbot_name": {"$exists": True}})}
    return human_projects.intersection(modified_projects)

    
def ionbot_first_to_prot_quant(df, lengths):
        """Only search in the rows : q-value <= 0.01, database = T, proteins != CONTAMINANT and '|'  
        Return a list of uniprot id"""
        
        df = df[['proteins', 'database', 'q-value']]
        df = df[(df['database']=='T') & (df['q-value']<=0.01)]
        df = df[~df['proteins'].apply(lambda x: '|' in x or 'CONTAMINANT' in x)]
        uniprot_df = df['proteins'].apply(lambda uniprot: re.search(r'\)\)\(\((.*?)\)\)', uniprot).group(1) 
                                          if re.search(r'\)\)\(\((.*?)\)\)', uniprot) else None) # Found the uniprot id with a ))((.*)) pattern
        uniprot_df = uniprot_df.value_counts().to_frame().reset_index()
        uniprot_df = uniprot_df.merge(lengths, left_on='proteins', right_on='Entry')
        uniprot_df = uniprot_df[uniprot_df['Length'] != np.nan]
        uniprot_df['count'] = uniprot_df['count'].astype(float)
        uniprot_df['Length'] = uniprot_df['Length'].astype(float)
        uniprot_df['SAF'] = uniprot_df['count']/uniprot_df['Length']
        total_SAF = uniprot_df['SAF'].sum()
        uniprot_df['NSAF'] = uniprot_df['SAF']/total_SAF
        return uniprot_df

def MLMarker_Arnaud_quant(prot_list):
    data = uniprot_df.pivot_table(columns='proteins', values='NSAF', aggfunc='sum')
    data = data.fillna(0)
    test = MLMarker(data.iloc[0:1, :], binary=False)
    prediction = MLMarker.adjusted_shap_values_df(test, penalty_factor=0.5, n_preds=100)
    prediction = prediction.sum(axis=1).sort_values(ascending=False)
    df_predi = pd.DataFrame(prediction).transpose()
    return df_predi


if __name__ == "__main__":
    for pxd in get_human_projects(sys.argv[1]):
        result_files = csv_path(f"/public/conode*/PRIDE_DATA/{pxd}/IONBOT_v0.11.3")
        lengths = pd.read_csv('./uniprot_reviewed_lengths.tsv', sep='/t')
        df = MLMarker_to_Dataframe(result_files, lengths)
        df.to_csv(os.path.join(os.path.join(sys.argv[2], f"{pxd}.csv")), header=True, index=False)
