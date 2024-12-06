import pandas as pd
import mysql.connector

class Atlas:
    def __init__(self, db, tissue, percentage, disease_status):
        self.db = db
        self.percentage = percentage
        self.conn = mysql.connector.connect(user='root', password='password', host='127.0.0.1', port='3306', database=self.db)
        if tissue in ('tissue_name', 'cell_type'):
            self.tissue = tissue
        else:
            print('Tissue type not defined')
            raise ValueError
        self.disease_status = disease_status

    def check_connection(self):
        if self.conn.is_connected():
            print("Connection successful")
        else:
            print("No connection")

    def get_pepseq(self):
        # Retrieve protein sequences from the database
        seqsql = "SELECT uniprot_id, length FROM protein WHERE length IS NOT NULL"
        seqData = pd.read_sql_query(seqsql, self.conn)
        seqData['length'] = pd.to_numeric(seqData['length'], errors='coerce')
        self.seqData = seqData
        return self.seqData

    def get_tissue_data(self):
        # Retrieve tissue data from the database
        tissuesql = "SELECT tissue_id, tissue_name, cell_type, disease_status FROM tissue"
        tissueData = pd.read_sql_query(tissuesql, self.conn)
        self.tissueData = tissueData
        return self.tissueData

    def get_assay_data(self):
        # Retrieve assay data from the database
        assaysql = "SELECT assay_id, peptide_id, quantification FROM peptide_to_assay"
        assayData = pd.read_sql_query(assaysql, self.conn)
        self.assayData = assayData
        return self.assayData

    def get_assay_tissue_data(self):
        # Retrieve assay-tissue mapping data from the database
        assaytissuesql = "SELECT assay_id, tissue_id FROM tissue_to_assay"
        assaytissueData = pd.read_sql_query(assaytissuesql, self.conn)
        self.assaytissueData = assaytissueData
        return self.assaytissueData

    def get_pep_data(self):
        # Retrieve peptide to protein mapping data from the database
        pepsql = "SELECT peptide_to_protein.peptide_id, peptide_to_protein.uniprot_id FROM peptide_to_protein"
        pepData = pd.read_sql_query(pepsql, self.conn)
        self.pepData = pepData
        return self.pepData

    def get_filtered_proteins(self):
        # Filter proteins based on specific criteria
        pepData = self.get_pep_data()
        proteotypicData = pepData.groupby("peptide_id").filter(lambda x: len(x) == 1)
        proteins = proteotypicData.groupby("uniprot_id").filter(lambda x: len(x) > 2)
        non_human_proteins = ['TRYP_PIG', 'TRY2_BOVIN', 'TRY1_BOVIN', 'SSPA_STAAU', 'SRPP_HEVBR', 'REF_HEVBR', 'ADH1_YEAST', 'ALBU_BOVIN', 'CAS1_BOVIN', 'CAS2_BOVIN', 'CASK_BOVIN', 'CASB_BOVIN', 'OVAL_CHICK', 'ALDOA_RABIT', 'BGAL_ECOLI', 'CAH2_BOVIN', 'CTRA_BOVIN', 'CTRB_BOVIN', 'CYC_HORSE', 'DHE3_BOVIN', 'GAG_SCVLA', 'GFP_AEQVI', 'K1C15_SHEEP', 'K1M1_SHEEP', 'K2M2_SHEEP', 'K2M3_SHEEP', 'KRA3A_SHEEP', 'KRA3_SHEEP', 'KRA61_SHEEP', 'LALBA_BOVIN', 'LYSC_CHICK', 'LYSC_LYSEN', 'MYG_HORSE', 'K1M2_SHEEP', 'K2M1_SHEEP']
        proteins = proteins[~proteins['uniprot_id'].isin(non_human_proteins)]
        self.proteins = proteins
        return self.proteins

    def get_protein_data(self):
        # Merge protein, tissue, and assay data
        seqData = self.get_pepseq()
        tissueData = self.get_tissue_data()
        assayData = self.get_assay_data()
        assaytissueData = self.get_assay_tissue_data()
        tissue_assay = pd.merge(assaytissueData, tissueData, on='tissue_id', how='left')
        tissue_assay = pd.merge(assayData, tissue_assay, on='assay_id', how='left')
        proteins = self.get_filtered_proteins()
        protData = pd.merge(tissue_assay, proteins, on='peptide_id').sort_values(['assay_id', 'uniprot_id'])
        if self.tissue == 'tissue_name':
            del protData['cell_type']
        elif self.tissue == 'cell_type':
            del protData['tissue_name']
        del protData['peptide_id']
        del protData['tissue_id']
        del protData['disease_status']
        self.protData = protData
        return self.protData

    def filter_protein_data(self):
        # Filter protein data based on a percentage threshold
        protData = self.get_protein_data()
        assays = protData[self.tissue].unique()
        DataFrameDict = {elem: pd.DataFrame for elem in assays}
        reduction = []
        for key in DataFrameDict.keys():
            DataFrameDict[key] = protData[protData[self.tissue] == key]
            perc = self.percentage * len(pd.unique(DataFrameDict[key]['assay_id']))
            before = DataFrameDict[key]['uniprot_id'].nunique()
            DataFrameDict[key] = DataFrameDict[key].groupby('uniprot_id').filter(lambda x: len(x) > perc)
            after = DataFrameDict[key]['uniprot_id'].nunique()
            reduction.append(before - after)
        filteredData = pd.DataFrame()
        for key in DataFrameDict.keys():
            filteredData = filteredData.append(DataFrameDict[key])
        del filteredData[self.tissue]
        self.filteredData = filteredData
        self.reduction = sum(reduction)
        return self.filteredData

    def calculate_NSAF(self):
        # Calculate NSAF scores for proteins
        filteredData = self.filter_protein_data()
        assays = filteredData['assay_id'].unique()
        DataFrameDict3 = {elem: pd.DataFrame for elem in assays}
        for key in DataFrameDict3.keys():
            DataFrameDict3[key] = filteredData[filteredData['assay_id'] == key]
        for key in DataFrameDict3.keys():
            sumSaf = 0
            assay = DataFrameDict3[key]
            assay.pop('assay_id')
            grouped = DataFrameDict3[key].groupby('uniprot_id').sum().reset_index()
            seqAddedDF = pd.merge(grouped, self.seqData, on='uniprot_id')
            seqAddedDF['SAF'] = seqAddedDF['quantification'] / seqAddedDF['length']
            sumSaf = seqAddedDF['SAF'].sum()
            seqAddedDF['NSAF'] = seqAddedDF['SAF'] / sumSaf
            del seqAddedDF['length']
            del seqAddedDF['quantification']
            del seqAddedDF['SAF']
            seqAddedDF.insert(loc=0, column='assay_id', value=key)
            DataFrameDict3[key] = seqAddedDF
        proteinData = pd.DataFrame()
        for key in DataFrameDict3.keys():
            proteinData = proteinData.append(DataFrameDict3[key])
        self.proteinData = proteinData
        return self.proteinData

    def get_predictor_atlas(self):
        # Generate the predictor atlas
        proteinData = self.calculate_NSAF()
        tissueData = self.get_tissue_data()
        if self.disease_status == 'Healthy':
            tissueData = tissueData[tissueData['disease_status'] == "Healthy"]
        self.atlas = pd.merge(proteinData, tissueData, on='assay_id')
        self.atlas = pd.pivot_table(self.atlas, values='NSAF', index='uniprot_id', columns='tissue_name').fillna(0)
        return self.atlas
