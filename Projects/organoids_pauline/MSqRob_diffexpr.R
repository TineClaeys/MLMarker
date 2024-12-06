library(tidyverse)
library(limma)
library(QFeatures)
library(msqrob2)
library(plotly)
library(gridExtra)

# Load the data from the two proteochip files
data_d0 <- read.csv("/home/compomics/git/MLMarker/Projects/organoids_pauline/proteochip_D0_report_organoids_culture_15042024_diann181.pg_matrix.tsv")
data_pfa <- read.csv("/home/compomics/git/MLMarker/Projects/organoids_pauline/proteochip_PFA_report_organoides_042024_fixes.pg_matrix.tsv")

# Perform differential expression analysis
result <- msqrob(data_D0, data_PFA)

