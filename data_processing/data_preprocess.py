# %%
import pandas as pd
import numpy as np

df = pd.read_csv('../data/msk_2024_mutations_clean.csv')

gene_counts = df['Gene'].value_counts(normalize=True)
genes_to_keep = gene_counts[gene_counts >= 0.001].index
df = df[df['Gene'].isin(genes_to_keep)]
print(f'Genes kept: {len(genes_to_keep)}')

genes = df["Gene"].unique()
# mutation subtypes
df["Mutation_Subtype"] = df["Gene"] + "_" + df["mutationType"] + "_" + df["variantType"] + "_chr" + df["chr"].astype(str)
mutation_subtypes = df["Mutation_Subtype"].unique()

# create a new df where each row is a patient and each column is a mutation subtype
patients = df["Patient"].unique()
patients_df = pd.DataFrame(index=patients, columns= genes.tolist() + mutation_subtypes.tolist())
patients_df = patients_df.fillna(0)

# Using a loop to fill in the DataFrame
for i, row in df.iterrows():
    patient = row["Patient"]
    mutation_subtype = row["Mutation_Subtype"]
    patients_df.loc[patient, mutation_subtype] = 1
    
    genes = row["Gene"]
    patients_df.loc[patient, genes] = 1
    
# add two columns to the DataFrame (overall survival and vital status)
patients_df["Overall_Survival_Months"] = np.nan
patients_df["Vital_Status"] = np.nan

# Check the result
print(patients_df.head())

# save the DataFrame to a csv file
patients_df.to_csv('../data/msk_2024_mutations__matrix.csv')



# %%
# OS_MONTHS = 'Overall_Survival_Months'
# OS_STATUS = 'Vital_Status'
unique_patients = patients_df.index.unique()
patients_mapping = {}

for patient in unique_patients:
    patient_data = cb.getAllClinicalDataOfPatientInStudy(studyId='msk_chord_2024', patientId=patient)
    temp = {}
    for data in patient_data:
        if data['clinicalAttributeId'] == 'OS_MONTHS':
            temp['OS_MONTHS'] = data['value']
        if data['clinicalAttributeId'] == 'OS_STATUS':
            temp['OS_STATUS'] = data['value']
    patients_mapping[patient] = temp
        
patients_mapping

with open('../data/patients_mapping.csv', 'w') as f:
    f.write("Patient,OS_MONTHS,OS_STATUS\n")
    for key in patients_mapping.keys():
        f.write(f"{key},{patients_mapping[key]['OS_MONTHS']},{patients_mapping[key]['OS_STATUS']}\n")