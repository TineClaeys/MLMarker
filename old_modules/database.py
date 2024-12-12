import pandas as pd
import mysql.connector
from collections import defaultdict
import glob
import os
import logging

class Database:
    def __init__(self, db_name='expression_atlas2', user='root', password='password', host='127.0.0.1', port='3306'):
        self.conn = mysql.connector.connect(user=user, password=password, host=host, port=port, database=db_name)
        self.mycursor = self.conn.cursor(buffered=True)
        # Check the connection
        if self.conn.is_connected():
            print("Connection successful")
        else:
            print("No connection")

    def check_projects(self, new_projects):
        # Check if a project is already in the DB. Returns empty list if no duplicates have been found
        for x in new_projects:
            if x[0:3] != 'PXD':
                print('Project name must begin with "PXD"')
                return False
        
        query = "SELECT PXD_accession FROM project"
        old_projects = pd.read_sql_query(query, self.conn)['PXD_accession'].values.tolist()
        duplicates = []
        for p in new_projects:
            if p in old_projects:
                print(f'{p} is already in the database')
                duplicates.append(p)
        print('Projects checked.')
        return duplicates

    def build_project_table(self, meta_df, list_of_pxds):
        # Populate project table using a dataframe with the necessary metadata and a list of PXD accessions
        meta_df = meta_df[meta_df['accession'].isin(list_of_pxds)]
        check = self.check_projects(meta_df.accession.unique().tolist())
        if check:
            print(f"Duplicates detected: {check}. \nNo entries have been added")
            return

        meta_df = meta_df[['accession', 'experimentTypes', 'instrumentNames', 'keywords', 'references']].astype(str)
        meta_tuples = list(meta_df.to_records(index=False))
        for i in meta_tuples:
            project = "INSERT INTO project(PXD_accession, experiment_type, instrument, keywords, ref) VALUES (%s, %s, %s, %s, %s)"
            self.mycursor.execute(project, list(i))
            self.conn.commit()
        print(f"{len(meta_tuples)} projects added to table 'project'.")

    def build_mod_table(self, mod_df):
        # Insert modifications into the modifications table
        mod_tuples = list(mod_df.to_records(index=False))
        for i in mod_tuples:
            mod = "INSERT IGNORE INTO modifications(mod_id, modification_type, mass_difference) VALUES(%s, %s, %s)"
            self.mycursor.execute(mod, list(i))
            self.conn.commit()
        print(f"{len(mod_tuples)} modifications added.")

    def build_tissue_table(self, tissue_df):
        # Insert tissues into the tissue table
        tissue_df = tissue_df[['tissue', 'cell_type', 'status']].drop_duplicates()
        tissue_tuples = list(tissue_df.to_records(index=False))
        for i in tissue_tuples:
            tissue = "INSERT INTO tissue(tissue_name, cell_type, disease_status) VALUES (%s, %s, %s)"
            self.mycursor.execute(tissue, list(i))
            self.conn.commit()
        print(f"{len(tissue_tuples)} tissues added.")

    def build_assay_cell_table(self, assay_df):
        # Insert assays into the assay table and link them with tissues
        assay_tuples = list(assay_df.to_records(index=False))
        for i in assay_tuples:
            accession, filename, pride_tissue, cell_type, tissue, status = i
            self.mycursor.execute("SELECT project_id FROM project WHERE PXD_accession = %s", (accession,))
            projectID = self.mycursor.fetchone()[0]
            assay = "INSERT INTO assay(project_id, filename) VALUES(%s, %s)"
            self.mycursor.execute(assay, (projectID, filename))
            self.conn.commit()
            assayID = self.mycursor.lastrowid
            self.mycursor.execute("SELECT tissue_id FROM tissue WHERE tissue_name = %s AND cell_type = %s AND disease_status = %s", (tissue, cell_type, status))
            tissueID = self.mycursor.fetchone()[0]
            tissue_to_assay = "INSERT INTO tissue_to_assay(assay_id, tissue_id) VALUES(%s, %s)"
            self.mycursor.execute(tissue_to_assay, (assayID, tissueID))
            self.conn.commit()
        print(f"{len(assay_tuples)} assays added.")

    def ionbot_parse(self, file):
        # Parse ionbot output files and filter based on specific criteria
        df = pd.read_csv(file, sep=',')
        if df.empty:
            logging.debug(f"File {file} is empty")
            return False
        df = df[(df['best_psm'] == 1) & (df['q_value'] <= 0.01) & (df['DB'] == 'T')]
        if df.empty:
            logging.debug(f"{file} did not pass filtering")
            return False
        df = df[~df['proteins'].str.contains('||', regex=False)]
        df['modifications'] = df['modifications'].fillna('x|[2030]unmodified')
        if df.empty:
            logging.debug(f"{file} all peptides are linked to multiple proteins or do not pass the filtering")
            return False
        spectral_counts = defaultdict(int)
        for pep in df['matched_peptide'].tolist():
            spectral_counts[pep] += 1
        spectral_counts = dict(sorted(spectral_counts.items(), key=lambda item: item[1], reverse=True))
        return df, spectral_counts

    def ionbot_store(self, file, filename):
        # Store parsed ionbot data into the database
        filename = filename.split('/')[-1].split('.')[0]
        self.mycursor.execute("SELECT assay_id FROM assay WHERE filename = %s", (filename,))
        assayID = self.mycursor.fetchone()
        if not assayID:
            print(f'{filename} is not in assays')
            return
        assayID = assayID[0]
        parser = self.ionbot_parse(file)
        if not parser:
            logging.warning(f"parser failed for {filename}")
            return
        df, spectral_counts = parser
        for _, row in df.iterrows():
            protID, pepseq, mod = row['proteins'], row['matched_peptide'], row['modifications']
            self.mycursor.execute("INSERT INTO peptide(peptide_sequence) VALUES (%s) ON DUPLICATE KEY UPDATE peptide_sequence=peptide_sequence", (pepseq,))
            self.conn.commit()
            self.mycursor.execute("SELECT peptide_id FROM peptide WHERE peptide_sequence = %s", (pepseq,))
            pepID = self.mycursor.fetchone()[0]
            uniprotID = protID.split('|')[1]
            self.mycursor.execute("INSERT INTO protein(uniprot_id) VALUES (%s) ON DUPLICATE KEY UPDATE uniprot_id=uniprot_id", (uniprotID,))
            self.conn.commit()
            self.mycursor.execute("INSERT INTO peptide_to_protein(uniprot_id, peptide_id) VALUES (%s,%s) ON DUPLICATE KEY UPDATE peptide_id=peptide_id, uniprot_id=uniprot_id", (uniprotID, pepID))
            self.conn.commit()
            for m in mod.split(';'):
                location, modID = m.split('|')[0], m[m.find("[")+1:m.find("]")]
                self.mycursor.execute("SELECT mod_id FROM modifications WHERE mod_id = %s", (modID,))
                modID = self.mycursor.fetchone()
                if modID:
                    modID = modID[0]
                    self.mycursor.execute("INSERT INTO peptide_modifications(peptide_id, location, mod_id, assay_id) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE count = count + 1", (pepID, location, modID, assayID))
                    self.conn.commit()
            count = spectral_counts.get(pepseq, float('inf'))
            self.mycursor.execute("INSERT INTO peptide_to_assay(peptide_id, assay_id, quantification) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE quantification=%s", (pepID, assayID, count, count))
            self.conn.commit()
        logging.info(f'{filename} was stored')

    def find_ionbot_files(self, projects):
        # Find and process ionbot files for given projects
        logging.basicConfig(filename='ionbot_assays.log', level=logging.DEBUG)
        number_of_files = 0
        for pxd in projects:
            path = None
            for base in ['/home/compomics/conode53_pride/PRIDE_DATA/', '/home/compomics/conode54_pride/PRIDE_DATA/', '/home/compomics/conode55_pride/PRIDE_DATA/']:
                if os.path.exists(base + str(pxd)):
                    path = base + str(pxd)
                    break
            if not path:
                continue
            for file in glob.glob(path + "/*.mgf.gzip/*.mgf.gzip.ionbot.csv"):
                number_of_files += 1
                if file not in logging.root.manager.loggerDict:
                    if os.path.getsize(file) != 0:
                        self.ionbot_store(file, file)
        print(number_of_files)
