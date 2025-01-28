import pandas as pd
import requests
import requests_cache
import numpy as np
import itertools
import yaml
from yaml import SafeLoader, load
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import luigi
from tqdm import tqdm

from pkb.config import INTERMEDIATE_DIR, OUTPUT_DIR, logger
from pkb.tasks.base import BaseTask
from pkb.tasks.harvard import HarvardIndexCollectorsTask
from pkb.tasks.wikidata import WikiDataCollectorsTask
from pkb.tasks.bionomia import BionomiaCollectorsTask

from Levenshtein import ratio

tqdm.pandas()

         
class CollectorsTask(BaseTask):
    """
    Look up index herbarium code
    """
    
    def requires(self):
        return [
            BionomiaCollectorsTask(),
            WikiDataCollectorsTask(),
            HarvardIndexCollectorsTask()
        ]

    # For all
    # Remove square blankets auto generated during data alignment process
    def clean_text(text): # fb
        text = text.replace('[', '').replace(']','').replace("'", '').replace("{", '').replace("}", '')
        return text

    def remove_spec_in_col(df, col):
        newCol = []
        for index, rowValue in df[col].items():
            if pd.notnull(rowValue):
                newCol.append(clean_text(rowValue))
            else:
                newCol.append(np.nan)
        return newCol
    
    def combine_name_list(df, colsList):
        nameList = []
        nameList = df[colsList].apply(lambda row: ','.join(row.dropna().unique()), axis=1)
        return nameList
    
    # For Wikidata
    
    def get_year(date_str):
        # Remove + sign
        if date_str[0] == '+':
            date_str = date_str[1:]
        return int(date_str[0:4])
    
    def convert_date2year(df, col):
        newCol = []
        for index, rowValue in df[col].items():
            if pd.notnull(rowValue):
                newCol.append(get_year(rowValue))
            else:
                newCol.append(np.nan)
        return newCol
    
    # For Bionomia
    
    def get_lifespan_year(text):
        if pd.notnull(text):
            # Replace special characters and split by &ndash;
            text = text.replace('&#42;', '').replace('&ndash;', '|').replace("&dagger;", '')
            parts = text.split('|')  # Split by &ndash; (now '|')
    
            # Initialize birth year and death year
            birth_year = None
            death_year = None
    
            # Extract years from each part
            if len(parts) == 2:  # If there are two parts (birth and death)
                birth_year = re.search(r'\b\d{4}\b', parts[0])  # Look for 4-digit year in the first part
                death_year = re.search(r'\b\d{4}\b', parts[1])  # Look for 4-digit year in the second part
    
                # Extract the year or leave as None if no match
                birth_year = birth_year.group(0) if birth_year else None
                death_year = death_year.group(0) if death_year else None
    
            elif len(parts) == 1:  # If only one part exists (e.g., no &ndash;)
                single_year = re.search(r'\b\d{4}\b', parts[0])
                birth_year = single_year.group(0) if single_year else None
    
            # Return list of birth and death years
            return [birth_year, death_year]
        else:
            return [None, None]
    
    def get_DoB_DoD(lifespan_list):
        if not lifespan_list or all(x is None or x is np.nan for x in lifespan_list):
            return [np.nan, np.nan]
        return lifespan_list
    
    def get_lifespan_DoB_DoD(df, col):
        birth_years = []
        death_years = []
        for index, row_value in df[col].items():
            temp = get_DoB_DoD(get_lifespan_year(row_value))
            birth_years.append(temp[0])
            death_years.append(temp[1])
        return birth_years, death_years
    
    def clean_accepted_names(accepted_names):
        # Step 1: Extract content inside square brackets
        inside_brackets = re.findall(r"\[.*?\]", accepted_names)
        if inside_brackets:
            # Clean and ensure proper separation within brackets
            inside_content = inside_brackets[0].strip('[]')  # Remove the outer brackets
            # Replace '\n' with a space and add commas between items separated only by spaces
            inside_cleaned = re.sub(r"\s{2,}", " ", inside_content.replace('\n', ' '))
            inside_cleaned = re.sub(r"'\s+'", "', '", inside_cleaned)  # Fix missing commas
            inside_cleaned = inside_cleaned.replace("'", '').strip()  # Remove single quotes
            # Replace the original bracketed content with cleaned content
            accepted_names = accepted_names.replace(inside_brackets[0], inside_cleaned)
        
        # Step 2: Remove remaining brackets, quotes, and extra spaces
        cleaned = accepted_names.replace('[', '').replace(']', '').replace("' '", ',').strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Replace multiple spaces with a single space
        
        # Step 3: Split by commas and clean individual parts
        parts = [part.strip() for part in cleaned.split(',')]
        cleaned = ', '.join(filter(bool, parts))  # Remove empty parts and join with commas
        
        return cleaned.replace("' '",", ").replace("'","")

    # For Harvard Index
    # Functions to extract the herbarium institution codes from Remarks in Harvard Index
    def get_herbarium_codes(string):
        herbarium_codes = []
        for s in string.split(","):
            if s.isupper():
                herbarium_codes.append(re.sub('[^A-Z]', ',', s).replace(",",""))
        return herbarium_codes
                
    def get_author_notes(string):
        authorNotes = []
        for s in string:
            # s = clean_text(s)
            # authorNotes.append(s.partition("author note: ")[2].partition(" ")[0].replace(";", ''))
            authorNotes.append(s.partition("author note: ")[2].partition("]")[0].replace(";", ',').replace(":", ',').replace("(",",").replace(")",",").replace("at",","))
        authorNotes = list(filter(None, authorNotes))
        return authorNotes
    
    def get_collector_notes(string):
        collectorNotes = []
        for s in string:
            # s = clean_text(s)
            # collectorNotes.append(s.partition("collector note: ")[2].partition(" ")[0].replace(";", ''))
            collectorNotes.append(s.partition("collector note: ")[2].partition("]")[0].replace(";", ',').replace(":", ',').replace("(",",").replace(")",",").replace("at",","))
        collectorNotes = list(filter(None, collectorNotes))
        return collectorNotes
    
    def get_author_collector_notes(df, col):
        authorNoteCol = []
        collectorNoteCol = []
        for index, rowValue in df[col].items():
            if pd.notnull(rowValue):
                string = rowValue.split("[")
                authorNoteCol.append(get_author_notes(string))
                collectorNoteCol.append(get_collector_notes(string))
            else:
                authorNoteCol.append([]) # use empty to reduce runtime exception while condition checking
                collectorNoteCol.append([])
        return authorNoteCol, collectorNoteCol
    
    def extract_herbariums(df, col):
        newCol = []
        for index, rowValue in df[col].items():
            if rowValue:
                temp = []
                # for i in rowValue: temp = get_herbarium_codes(i)
                for i in rowValue: 
                    temp += get_herbarium_codes(i)
                newCol.append(temp)
            else:
                newCol.append([])
        return newCol
    
    # Replace empty list as np.nan
    def remove_empty(df, col):
        newCol = []
        for index, rowValue in df[col].items():
            if not rowValue:
                newCol.append(np.nan)
            else:
                newCol.append(rowValue)
        return newCol

    def merge_wiki_geo(df):
        def clean_entry(entry):
            # Handle sets: convert to strings and clean 'None'
            if isinstance(entry, set):
                # Remove 'None' from sets and join remaining elements
                cleaned_set = {x for x in entry if x != 'None'}
                return ', '.join(cleaned_set) if cleaned_set else np.nan
        
            # Handle strings: clean multiple occurrences of 'None'
            elif isinstance(entry, str):
                # Split the string into parts, remove 'None', and rejoin
                cleaned = ', '.join(part.strip() for part in entry.split(',') if part.strip() != 'None')
                return cleaned if cleaned else np.nan  # Replace empty results with NaN
        
            # Return NaN for other invalid types (e.g., float NaN)
            else:
                return entry
    
        # Clean both columns
        df['countriesOfCitizenship'] = df['countriesOfCitizenship'].apply(clean_entry)
        df['employerCountries'] = df['employerCountries'].apply(clean_entry)
    
        # Merge columns, prioritizing countriesOfCitizenship and filling with employerCountries
        return df['countriesOfCitizenship'].combine_first(df['employerCountries'])


    # Function to check the condition
    def is_valid_date_pair_WH(wikiID, harvardIndex):
        dob_dfa = dfa.at[wikiID, 'dateOfBirth']
        dob_dfb = dfb.at[harvardIndex, 'dateOfBirth']
        dod_dfa = dfa.at[wikiID, 'dateOfDeath']
        dod_dfb = dfb.at[harvardIndex, 'dateOfDeath']
        
        result = True
    
        # Check if both dateOfBirth are not NaN and if they are different
        if pd.notna(dod_dfa) and pd.notna(dod_dfb) and abs(dod_dfa - dod_dfb) > 5:
            result = False
        
        if pd.notna(dob_dfa) and pd.notna(dob_dfb) and abs(dob_dfa - dob_dfb) > 5:
            result = False
        
        if pd.notna(dob_dfa) and pd.notna(dob_dfb) and pd.notna(dod_dfa) and pd.notna(dod_dfb):
            if abs(dob_dfa - dob_dfb) > 5 and abs(dob_dfa - dob_dfb) <= 15 and dod_dfa == dod_dfb:
                result = True
            if dob_dfa == dob_dfb and abs(dod_dfa - dod_dfb) > 5 and abs(dod_dfa - dod_dfb) <= 15:
                result = True
              
        if pd.notna(dob_dfa) and pd.notna(dod_dfa) and dod_dfa == dob_dfb:
            result = True
        
        return result
        
    
    def generate_match_pairs_validated(dataset1, dataset2, lastName_threshold, firstName_threshold):  
    
        # Initialize indexer
        indexer = recordlinkage.Index()
    
        # Sorted neighborhood indexing on lastname
        indexer.sortedneighbourhood('lastName', window=3)
        
        # Blocking on author abbreviation - generate index pairs if 'exact match on author abbreviation'
        indexer.block(left_on = 'authorAbbrv', right_on='authorAbbrv')
    
        candidate_links = indexer.index(dataset1, dataset2)
    
        compare_cl = recordlinkage.Compare()
    
        # Exact match on author abbreviation, extra conditioning on the blocking
        compare_cl.exact('authorAbbrv', 'authorAbbrv', label='authorAbbrv')
        
        # Compare last names with Levenshtein distance
        # When threshold set to none, the feature table stores the similarity percentage calculated using Levenshtein distance
        compare_cl.string('lastName', 'lastName', method='damerau_levenshtein', threshold=None, label='lastName')
    
        # Compare first names with Levenshtein distance
        compare_cl.string('firstName', 'firstName', method='damerau_levenshtein', threshold=None, label='firstName')
        
        # Exact match on first name initial
        compare_cl.exact('firstName_initial', 'firstName_initial',label='firstName_initial')
    
        # Exact match on first name initial from label(normalised full name)
        compare_cl.exact('first_name_initial', 'first_name_initial', label='first_name_initial')
    
        # Exact match on date of birth
        compare_cl.exact('dateOfBirth', 'dateOfBirth', label='dateOfBirth')
    
        # Exact match on date of death
        compare_cl.exact('dateOfDeath', 'dateOfDeath', label='dateOfDeath')
    
        # Compute the comparison results
        features = compare_cl.compute(candidate_links, dataset1, dataset2)
        # print(len(features))
        
        # Filter pairs based on the given criteria
        index_pairs = []
        # Set the threshold of names_not_completely_different function (firstname check after initial comparison)
        threshold = 0.5
        # Set the threshold for doing aliases matching
        alias_thre = 0.85
    
        for index, row in features.iterrows():
            a_index = index[0]
            b_index = index[1]
            # Waterfall model to check criteria
            if row['lastName'] >= lastName_threshold:
                if row['firstName'] >= firstName_threshold:
                    index_pairs.append(index)
                if row['firstName_initial'] == 1:
                    firstName1 = dataset1.at[index[0], 'firstName']
                    firstName2 = dataset2.at[index[1], 'firstName']
                    if not (is_initial(firstName1) or is_initial(firstName2)) and names_not_completely_different(firstName1, firstName2, threshold):
                        index_pairs.append(index)
                if row['first_name_initial'] == 1:
                    if not (is_initial(firstName1) or is_initial(firstName2)) and names_not_completely_different(firstName1, firstName2, threshold):
                        index_pairs.append(index)
                        
        # Remove duplicate pairs
        index_pairs = list(set(index_pairs))
        index_pairs = [pair for pair in index_pairs if is_valid_date_pair_WH(pair[0], pair[1])]
        
        for index, row in features.iterrows():
            a_index = index[0]
            b_index = index[1]
            # Waterfall model to check criteria
            if row['lastName'] >= lastName_threshold:
                if row['firstName_initial'] == 1 and row['dateOfBirth'] == 1:
                    index_pairs.append(index)
                if row['first_name_initial'] == 1 and row['dateOfBirth'] == 1:
                    index_pairs.append(index)
                if row['dateOfBirth'] == 1 and row['dateOfDeath'] == 1:
                    index_pairs.append(index)
                # Check if b's label is in a's aliases list
                if pd.notna(dataset1.at[a_index, 'aliases']) and (row['dateOfBirth'] == 1 or row['dateOfDeath'] == 1):
                    aliases_list = [alias.strip() for alias in dataset1.at[a_index, 'aliases'].split(',')]
                    for alias in aliases_list:
                        if pd.notna(dataset2.at[b_index, 'label']) and pd.notna(alias) and ratio(dataset2.at[b_index, 'label'], alias) >= alias_thre:
                            index_pairs.append(index)
                            break              
                # Check if a's label is in b's aliases list
                if pd.notna(dataset2.at[b_index, 'Name']) and (row['dateOfBirth'] == 1 or row['dateOfDeath'] == 1):
                    aliases_list = [alias.strip() for alias in dataset2.at[b_index, 'Name'].split(',')]
                    for alias in aliases_list:
                        if pd.notna(dataset1.at[a_index, 'name']) and pd.notna(alias) and ratio(dataset1.at[a_index, 'name'], alias) >= alias_thre:
                            index_pairs.append(index)
                            break
        
        for index, row in features.iterrows():        
            # Store the record pairs if their author abbreviations are the same
            if row['authorAbbrv'] == 1:
                index_pairs.append(index)
                
        # Remove duplicate pairs
        valid_index_pairs = list(set(index_pairs))
        
        # Convert the filtered pairs back to a MultiIndex
        multi_index = pd.MultiIndex.from_tuples(valid_index_pairs, names=["wikiID", "harvardIndex"])
    
        return multi_index
    
    
    # Function to check the condition
    def is_valid_date_pair_WB(wikiID, bioID):
        dob_dfa = dfa.at[wikiID, 'dateOfBirth']
        dob_dfc = dfc.at[bioID, 'dateOfBirth']
        dod_dfa = dfa.at[wikiID, 'dateOfDeath']
        dod_dfc = dfc.at[bioID, 'dateOfDeath']
        
        result = True
    
        # Check if both dateOfBirth are not NaN and if they are different
        if pd.notna(dod_dfa) and pd.notna(dod_dfc) and abs(dod_dfa - dod_dfc) > 5:
            result = False
        
        if pd.notna(dob_dfa) and pd.notna(dob_dfc) and abs(dob_dfa - dob_dfc) > 5:
            result = False
        
        if pd.notna(dob_dfa) and pd.notna(dob_dfc) and pd.notna(dod_dfa) and pd.notna(dod_dfc):
            if abs(dob_dfa - dob_dfc) > 5 and abs(dob_dfa - dob_dfc) <= 15 and dod_dfa == dod_dfc:
                result = True
            if dob_dfa == dob_dfc and abs(dod_dfa - dod_dfc) > 5 and abs(dod_dfa - dod_dfc) <= 15:
                result = True
                
        if pd.notna(dob_dfa) and pd.notna(dod_dfa) and dod_dfa == dob_dfc:
            result = True
        
        return result
    
    # Model to find the possible matching records between two datasets
    def generate_match_pairs_bio_validated(dataset1, dataset2, lastName_threshold, firstName_threshold):  
    
        # Initialize indexer
        indexer = recordlinkage.Index()
    
        # Sorted neighborhood indexing on lastname
        indexer.sortedneighbourhood('lastName', window=3)
    
        candidate_links = indexer.index(dataset1, dataset2)
        # print(len(candidate_links))
    
        compare_cl = recordlinkage.Compare()
        
        # Compare last names with Levenshtein distance
        # When threshold set to none, the feature table stores the similarity percentage calculated using Levenshtein distance
        compare_cl.string('lastName', 'lastName', method='damerau_levenshtein', threshold=None, label='lastName')
    
        # Compare first names with Levenshtein distance
        compare_cl.string('firstName', 'firstName', method='damerau_levenshtein', threshold=None, label='firstName')
        
        # Exact match on first name initial
        compare_cl.exact('firstName_initial', 'firstName_initial',label='firstName_initial')
    
        # Exact match on first name initial from label(normalised full name)
        compare_cl.exact('first_name_initial', 'first_name_initial', label='first_name_initial')
    
        # Exact match on date of birth
        compare_cl.exact('dateOfBirth', 'dateOfBirth', label='dateOfBirth')
    
        # Exact match on date of death
        compare_cl.exact('dateOfDeath', 'dateOfDeath', label='dateOfDeath')
    
        # Compute the comparison results
        features = compare_cl.compute(candidate_links, dataset1, dataset2)
        # print(len(features))
        
        # Filter pairs based on the given criteria
        index_pairs = []
        # Set the threshold of names_not_completely_different function (firstname check after initial comparison)
        threshold = 0.7
        # Set the threshold for doing aliases matching
        alias_thre = 0.9 
    
        for index, row in features.iterrows():
            a_index = index[0]
            c_index = index[1]
                
            # Waterfall model to check criteria
            if row['lastName'] >= lastName_threshold:
                if row['firstName'] >= firstName_threshold:
                    index_pairs.append(index)
                if row['firstName_initial'] == 1:
                    firstName1 = dataset1.at[a_index, 'firstName']
                    firstName2 = dataset2.at[c_index, 'firstName']
                    if not (is_initial(firstName1) or is_initial(firstName2)) and names_not_completely_different(firstName1, firstName2, threshold):
                        index_pairs.append(index)
                if row['first_name_initial'] == 1:
                    firstName1 = dataset1.at[a_index, 'firstName']
                    firstName2 = dataset2.at[c_index, 'firstName']
                    if not (is_initial(firstName1) or is_initial(firstName2)) and names_not_completely_different(firstName1, firstName2, threshold):
                        index_pairs.append(index)
                if row['firstName_initial'] == 1 and row['dateOfBirth'] == 1:
                    index_pairs.append(index)
                if row['first_name_initial'] == 1 and row['dateOfBirth'] == 1:
                    index_pairs.append(index)
                # if row['dateOfBirth'] == 1:
                #     index_pairs.append(index)
                if row['dateOfBirth'] == 1 and row['dateOfDeath'] == 1:
                    index_pairs.append(index)
                # Check if b's label is in a's aliases list
                if pd.notna(dataset1.at[a_index, 'aliases']) and row['dateOfBirth'] == 1:
                    aliases_list = [alias.strip() for alias in dataset1.at[a_index, 'aliases'].split(',')]
                    for alias in aliases_list:
                        if pd.notna(dataset2.at[c_index, 'label']) and pd.notna(alias) and ratio(dataset2.at[c_index, 'label'], alias) >= alias_thre:
                            index_pairs.append(index)
                            break              
                # Check if a's label is in b's aliases list
                if pd.notna(dataset2.at[c_index, 'acceptedNames']) and row['dateOfBirth'] == 1:
                    aliases_list = [alias.strip() for alias in dataset2.at[c_index, 'acceptedNames'].split(',')]
                    for alias in aliases_list:
                        if pd.notna(dataset1.at[a_index, 'name']) and pd.notna(alias) and ratio(dataset1.at[a_index, 'name'], alias) >= alias_thre:
                            index_pairs.append(index)
                            break
                if pd.notna(dataset1.at[a_index, 'aliases']) and row['dateOfBirth'] == 1:
                    aliases_list = [alias.strip() for alias in dataset1.at[a_index, 'aliases'].split(',')]
                    for alias in aliases_list:
                        if pd.notna(dataset2.at[c_index, 'label']) and pd.notna(alias) and ratio(dataset2.at[c_index, 'label'], alias) >= alias_thre:
                            index_pairs.append(index)
                            break
    
        # Remove duplicate pairs
        index_pairs = list(set(index_pairs))
        
        # Validation check based on conflict of dob and dod - reduce FP
        valid_index_pairs = [pair for pair in index_pairs if is_valid_date_pair_WB(pair[0], pair[1])]
        # Convert the filtered pairs back to a MultiIndex
        valid_index_pairs = pd.MultiIndex.from_tuples(valid_index_pairs, names=['wikiID', 'bioID'])
    
        return valid_index_pairs
    
    # Function to check the condition
    def is_valid_date_pair_HB(harvardIndex, bioID):
        dob_dfb = dfb.at[harvardIndex, 'dateOfBirth']
        dob_dfc = dfc.at[bioID, 'dateOfBirth']
        dod_dfb = dfb.at[harvardIndex, 'dateOfDeath']
        dod_dfc = dfc.at[bioID, 'dateOfDeath']
        
        result = True
    
        # Check if both dateOfBirth are not NaN and if they are different
        if pd.notna(dod_dfb) and pd.notna(dod_dfc) and abs(dod_dfb - dod_dfc) > 5:
            result = False
        
        if pd.notna(dob_dfb) and pd.notna(dob_dfc) and abs(dob_dfb - dob_dfc) > 5:
            result = False
        
        if pd.notna(dob_dfb) and pd.notna(dob_dfc) and pd.notna(dod_dfb) and pd.notna(dod_dfc):
            if abs(dob_dfb - dob_dfc) > 5 and abs(dob_dfb - dob_dfc) <= 15 and dod_dfb == dod_dfc:
                result = True
            if dob_dfb == dob_dfc and abs(dod_dfb - dod_dfc) > 5 and abs(dod_dfb - dod_dfc) <= 15:
                result = True
                
        if pd.notna(dob_dfb) and pd.notna(dod_dfb) and dod_dfb == dob_dfc:
            result = True
        
        return result
    
    # Model to find the possible matching records between two datasets
    def generate_match_pairs_Hbio_validated(dataset1, dataset2, lastName_threshold, firstName_threshold):  
    
        # Initialize indexer
        indexer = recordlinkage.Index()
    
        # Sorted neighborhood indexing on lastname
        indexer.sortedneighbourhood('lastName', window=3)
    
        candidate_links = indexer.index(dataset1, dataset2)
        # print(len(candidate_links))
    
        compare_cl = recordlinkage.Compare()
        
        # Compare last names with Levenshtein distance
        # When threshold set to none, the feature table stores the similarity percentage calculated using Levenshtein distance
        compare_cl.string('lastName', 'lastName', method='damerau_levenshtein', threshold=None, label='lastName')
    
        # Compare first names with Levenshtein distance
        compare_cl.string('firstName', 'firstName', method='damerau_levenshtein', threshold=None, label='firstName')
        
        # Exact match on first name initial
        compare_cl.exact('firstName_initial', 'firstName_initial',label='firstName_initial')
    
        # Exact match on first name initial from label(normalised full name)
        compare_cl.exact('first_name_initial', 'first_name_initial', label='first_name_initial')
    
        # Exact match on date of birth
        compare_cl.exact('dateOfBirth', 'dateOfBirth', label='dateOfBirth')
    
        # Exact match on date of death
        compare_cl.exact('dateOfDeath', 'dateOfDeath', label='dateOfDeath')
    
        # Compute the comparison results
        features = compare_cl.compute(candidate_links, dataset1, dataset2)
        # print(len(features))
        
        # Filter pairs based on the given criteria
        index_pairs = []
        # Set the threshold of names_not_completely_different function (firstname check after initial comparison)
        threshold = 0.7
        # Set the threshold for doing aliases matching
        alias_thre = 0.9 
    
        for index, row in features.iterrows():
            b_index = index[0]
            c_index = index[1]
                
            # Waterfall model to check criteria
            if row['lastName'] >= lastName_threshold:
                if row['firstName'] >= firstName_threshold:
                    index_pairs.append(index)
                if row['firstName_initial'] == 1:
                    firstName1 = dataset1.at[b_index, 'firstName']
                    firstName2 = dataset2.at[c_index, 'firstName']
                    if not (is_initial(firstName1) or is_initial(firstName2)) and names_not_completely_different(firstName1, firstName2, threshold):
                        index_pairs.append(index)
                if row['first_name_initial'] == 1:
                    firstName1 = dataset1.at[b_index, 'firstName']
                    firstName2 = dataset2.at[c_index, 'firstName']
                    if not (is_initial(firstName1) or is_initial(firstName2)) and names_not_completely_different(firstName1, firstName2, threshold):
                        index_pairs.append(index)
                if row['firstName_initial'] == 1 and row['dateOfBirth'] == 1:
                    index_pairs.append(index)
                if row['first_name_initial'] == 1 and row['dateOfBirth'] == 1:
                    index_pairs.append(index)
                # if row['dateOfBirth'] == 1:
                #     index_pairs.append(index)
                if row['dateOfBirth'] == 1 and row['dateOfDeath'] == 1:
                    index_pairs.append(index)
                # Check if b's label is in a's aliases list
                if pd.notna(dataset1.at[b_index, 'Name']) and row['dateOfBirth'] == 1:
                    aliases_list = [alias.strip() for alias in dataset1.at[b_index, 'Name'].split(',')]
                    for alias in aliases_list:
                        if pd.notna(dataset2.at[c_index, 'label']) and pd.notna(alias) and ratio(dataset2.at[c_index, 'label'], alias) >= alias_thre:
                            index_pairs.append(index)
                            break              
                # Check if a's label is in b's aliases list
                if pd.notna(dataset2.at[c_index, 'acceptedNames']) and row['dateOfBirth'] == 1:
                    aliases_list = [alias.strip() for alias in dataset2.at[c_index, 'acceptedNames'].split(',')]
                    for alias in aliases_list:
                        if pd.notna(dataset1.at[b_index, 'label']) and pd.notna(alias) and ratio(dataset1.at[b_index, 'label'], alias) >= alias_thre:
                            index_pairs.append(index)
                            break
                if pd.notna(dataset1.at[b_index, 'Name']) and row['dateOfBirth'] == 1:
                    aliases_list = [alias.strip() for alias in dataset1.at[b_index, 'Name'].split(',')]
                    for alias in aliases_list:
                        if pd.notna(dataset2.at[c_index, 'label']) and pd.notna(alias) and ratio(dataset2.at[c_index, 'label'], alias) >= alias_thre:
                            index_pairs.append(index)
                            break
    
        # Remove duplicate pairs
        index_pairs = list(set(index_pairs))
        
        # Validation check based on conflict of dob and dod - reduce FP
        valid_index_pairs = [pair for pair in index_pairs if is_valid_date_pair_HB(pair[0], pair[1])]
        # Convert the filtered pairs back to a MultiIndex
        valid_index_pairs = pd.MultiIndex.from_tuples(valid_index_pairs, names=['harvardIndex', 'bioID'])
    
        return valid_index_pairs
    
    # Define the function to split the label(full name) into first name and last name
    def split_full_name(full_name):
        if pd.isnull(full_name) or full_name == '':
            return '', ''
        parts = full_name.split()
        first_name = ' '.join(parts[:-1])
        last_name = parts[-1]
        return first_name, last_name
    
    # Define the function to convert each word in the first name to the desired format
    def convert_to_initial(name):
        if pd.isnull(name) or name == '':
            return ''
        initials = [word[0].upper() + '.' for word in name.split()]
        return ' '.join(initials)
    
    # Helper function to combine two multi index objects with the same index names
    def combine_multi_index(index1, index2):
        # Convert MultiIndex objects to DataFrames
        multi_index1 = index1.to_frame(index=False)
        multi_index2 = index2.to_frame(index=False)
    
        # Concatenate the DataFrames
        combined_index = pd.concat([multi_index1, multi_index2])
    
        # Remove duplicates
        combined_index = combined_index.drop_duplicates()
    
        # Convert back to MultiIndex
        combined_multi_index = pd.MultiIndex.from_frame(combined_index)
        
        return combined_multi_index
    
    def is_initial(name):
        if not isinstance(name, str):  # Ensure name is a string
            return False
        parts = name.split()
        return all(len(part) == 2 and part[0].isalpha() and part[1] == '.' for part in parts)
    
    def names_not_completely_different(name1, name2, threshold):
        return ratio(name1, name2) > threshold
    
    # This fuction is designed to find the id matched between wikidata and harvard index
    def define_true_pairs(indexList1, indexList2, indexName1, indexName2):
        arrays = [indexList1, indexList2]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples, names=[indexName1, indexName2])
        return index

    
    def run(self):
        
        # TODO: Add Hiris' collectors aggregation code
        wiki_file = '~/Documents/nhm_coding/pkbdata_2024/wiki-collectors.csv'  ## macOS file location, wikipedia collector
        wiki_data = pd.read_csv(wiki_file,chunksize=10000,encoding='utf-8',on_bad_lines='skip',engine='python')
        wiki_data = pd.concat(wiki_data)
        a = wiki_data[['qid','name','dateOfBirth','dateOfDeath','countriesOfCitizenship',
              'harvardIndex','bionomia','authorAbbrv','aliases','employers','employerCountries']].copy()
        a['firstName'] = a['name'].str.split('\s+').str[0]
        a['lastName'] = a['name'].str.split('\s+').str[-1]
        a['dateOfBirth'] = convert_date2year(a,'dateOfBirth')
        a['dateOfDeath'] = convert_date2year(a,'dateOfDeath')
        a['employers'] = remove_spec_in_col(a,'employers')
        a['aliases'] = remove_spec_in_col(a,'aliases')
        a['countriesOfCitizenship'] = remove_spec_in_col(a,'countriesOfCitizenship')
        a['employerCountries'] = remove_spec_in_col(a,'employerCountries')
        a['harvardIndex'] = pd.to_numeric(a['harvardIndex'],errors='coerce')
        a['countries'] = merge_wiki_geo(a)
        cols_to_drop = ['countriesOfCitizenship', 'employerCountries', 'employers']
        a.drop(cols_to_drop, axis=1, inplace=True)

        harvard_file = '~/Documents/nhm_coding/pkbdata_2024/harvard-collectors.csv'
        harvard_data = pd.read_csv(harvard_file,chunksize=10000,encoding='utf-8',on_bad_lines='skip',engine='python')
        harvard_data = pd.concat(harvard_data)
        b = harvard_data[['id','Standard/Label Name','birthYear','deathYear','birthYearIsApprox','geographyISO','herbariaCode',
                     'firstName','middleName','lastName','B & P Author Abbrev.','Name','Remarks']].copy()
        b.rename(columns={"birthYear": "dateOfBirth", "deathYear": "dateOfDeath", "Standard/Label Name":"label", 
                          "B & P Author Abbrev.":"authorAbbrv"}, inplace=True)
        b['Name'] = remove_spec_in_col(b,'Name')
        b['geographyISO'] = remove_spec_in_col(b,'geographyISO')
        b['herbariaCode'] = remove_spec_in_col(b,'herbariaCode')
        temp0, temp1 = get_author_collector_notes(b, 'Remarks')
        b['author note'] = temp0
        b['collector note'] = temp1
        b['author note'] = extract_herbariums(b,'author note')
        cols_to_drop = ['Remarks','collector note']
        b.drop(cols_to_drop, axis=1, inplace=True)

        bionomia_file = '~/Documents/nhm_coding/pkbdata_2024/bionomia-collectors.parquet'  ## macOS file location, bionomia collector
        bionomia_data = pd.read_parquet(bionomia_file)
        c = bionomia_data[['id','orcid','wikidata','fullname','fullname_reverse','label','given','family',
                           'orgs','countries','other_names','lifespan']].copy()
        dob, dod = get_lifespan_DoB_DoD(c, 'lifespan')
        c['dateOfBirth'] = dob
        c['dateOfDeath'] = dod
        c.rename(columns={"given":"firstName", "family":"lastName"}, inplace=True)
        # c['acceptedNames'] = c['other_names'].apply(lambda x: '' if isinstance(x, list) and not x else str(x)) \
        #                       + ', ' + c['fullname_reverse'].fillna('').astype(str).str.replace(',', '')
        c['acceptedNames'] = c['other_names'].apply(lambda x: '' if isinstance(x, list) and not x else str(x))
        c['acceptedNames'] = c['acceptedNames'].apply(clean_accepted_names)
        cols_to_drop = ['lifespan', 'fullname_reverse','other_names']
        c.drop(cols_to_drop, axis=1, inplace=True)

        # Preprocess and clean data
        a['wikiID'] = a['qid']
        a = a.set_index('qid')
        b['harvardIndex'] = b['id']
        b = b.set_index('id')
        c['bioID'] = c['id']
        c = c.set_index('id')
        
        dfa = a.copy()
        dfb = b.copy()
        dfc = c.copy()
        dfc['dateOfBirth'] = pd.to_numeric(dfc['dateOfBirth'], errors='coerce')
        dfc['dateOfDeath'] = pd.to_numeric(dfc['dateOfDeath'], errors='coerce')
        
        # Set indices
        dfa.set_index('wikiID', inplace=True)
        dfb.set_index('harvardIndex', inplace=True)
        dfc.set_index('bioID', inplace=True)
        
        # Apply the split_full_name function to separate first name and last name
        dfa[['first_name', 'last_name']] = dfa['name'].apply(lambda x: pd.Series(split_full_name(x)))
        # Apply the convert_to_initial function to the first name column
        dfa['first_name_initial'] = dfa['first_name'].apply(convert_to_initial)
        
        # Apply the split_full_name function to separate first name and last name
        dfb[['first_name', 'last_name']] = dfb['label'].apply(lambda x: pd.Series(split_full_name(x)))
        # Apply the convert_to_initial function to the first name column
        dfb['first_name_initial'] = dfb['first_name'].apply(convert_to_initial)
        
        # Apply the split_full_name function to separate first name and last name
        dfc[['first_name', 'last_name']] = dfc['label'].apply(lambda x: pd.Series(split_full_name(x)))
        # Apply the convert_to_initial function to the first name column
        dfc['first_name_initial'] = dfc['first_name'].apply(convert_to_initial)
        
        # Apply the convert_to_initial function to the first name column in Wiki
        dfa['firstName_initial'] = dfa['firstName'].apply(convert_to_initial)
        # Apply the convert_to_initial function to the first name column in HI
        dfb['firstName_initial'] = dfb['firstName'].apply(convert_to_initial)
        # Apply the convert_to_initial function to the first name column in Bionomia
        dfc['firstName_initial'] = dfc['firstName'].apply(convert_to_initial)
        
        # Drop duplicated last name column
        dfa.drop('last_name', axis=1, inplace=True)
        dfb.drop('last_name', axis=1, inplace=True)
        dfc.drop('last_name', axis=1, inplace=True)
        
        print('Finished data preparation for Wikidata, HI data and Bionomia data')
        
        # Find out the ground truth matches of wikidata and harvard index using id in the records
        temp = pd.merge(a, b, how='inner', on=None, left_on='harvardIndex', right_on='harvardIndex',
                          left_index=False, right_index=False, sort=False,
                          suffixes=('_wiki', '_harvard'), copy=False, indicator=False)
        
        true_matches_WH = define_true_pairs(temp['wikiID'],temp['harvardIndex'].astype(int),'wikiID','harvardIndex')
        # print("\nWikiID and havardIndex pairs as true matches:")
        # print(true_matches)
        
        # Print out the precentage
        print('There are '+ str(len(true_matches_WH)) +' HarvardIndex records in Wikidata, which is ' + str(len(true_matches_WH)/len(a)*100) +'%')
        print('There are '+ str(len(true_matches_WH)) +' Wikidata records in HarvardIndex, which is ' + str(len(true_matches_WH)/len(b)*100) +'%')
        
        # Find out the ground truth matches of wikidata and bionomia using id in the records
        temp = pd.merge(a, c, how='inner', on=None, left_on='wikiID', right_on='wikidata',
                          left_index=False, right_index=False, sort=False,
                          suffixes=('_wiki', '_bionomia'), copy=False, indicator=False)
        
        true_matches_WB = define_true_pairs(temp['wikiID'],temp['bioID'].astype(int),'wikiID','bioID')
        # print("\nWikiID and bionomia pairs as true matches:")
        # print(true_matches)
        
        # Print out the precentage
        print('There are '+ str(len(true_matches_WB)) +' Bionomia records in Wikidata, which is ' + str(len(true_matches_WB)/len(a)*100) +'%')
        print('There are '+ str(len(true_matches_WB)) +' Wikidata records in Bionomia, which is ' + str(len(true_matches_WB)/len(c)*100) +'%')
        
        # Finding all possible matches between wikidata and harvard index (all data)
        temp_index = generate_match_pairs_validated(dfa, dfb, lastName_threshold=0.9, firstName_threshold=0.85)
        # print(len(temp_index))
        
        # Combine the possible matches with the true matches generated by exact id matching
        index_pairs_WH_valid = combine_multi_index(temp_index, true_matches_WH)
        print("There are {} possible matches found based on the given criteria".format(len(index_pairs_WH_valid)))
        
        # Print out the precentage
        print('There are '+ str(len(index_pairs_WH_valid)) +' HarvardIndex records in Wikidata, which is ' + str(len(index_pairs_WH_valid)/len(a)*100) +'%')
        print('There are '+ str(len(index_pairs_WH_valid)) +' Wikidata records in HarvardIndex, which is ' + str(len(index_pairs_WH_valid)/len(b)*100) +'%')
        
        # Merge the matches with the original data
        matched_dfa = dfa.loc[index_pairs_WH_valid.get_level_values('wikiID')].reset_index()
        matched_dfb = dfb.loc[index_pairs_WH_valid.get_level_values('harvardIndex')].reset_index()
        
        # Combine the matched DataFrames side by side
        combined_matches_WH_valid = pd.concat([matched_dfa, matched_dfb], axis=1)
        
        # Optionally add a label to identify matched rows
        combined_matches_WH_valid['matched'] = True
        
        # Display the combined DataFrame
        # combined_matches_WH_valid.head()
        
        # Finding all possible matches between wikidata and bionomia (all data)
        temp_index = generate_match_pairs_bio_validated(dfa, dfc, lastName_threshold=0.9, firstName_threshold=0.85)
        # print(len(temp_index))
        
        # Combine the possible matches with the true matches generated by exact id matching
        index_pairs_WB_valid = combine_multi_index(temp_index, true_matches_WB)
        print("There are {} possible matches found based on the given criteria".format(len(index_pairs_WB_valid)))
        
        # Print out the precentage
        print('There are '+ str(len(index_pairs_WB_valid)) +' Bionomia records in Wikidata, which is ' + str(len(index_pairs_WB_valid)/len(a)*100) +'%')
        print('There are '+ str(len(index_pairs_WB_valid)) +' Wikidata records in Bionomia, which is ' + str(len(index_pairs_WB_valid)/len(c)*100) +'%')
        
        # Merge the matches with the original data
        matched_dfa = dfa.loc[index_pairs_WB_valid.get_level_values('wikiID')].reset_index()
        matched_dfc = dfc.loc[index_pairs_WB_valid.get_level_values('bioID')].reset_index()
        
        # Combine the matched DataFrames side by side
        combined_matches_WB_valid = pd.concat([matched_dfa, matched_dfc], axis=1)
        
        # Optionally add a label to identify matched rows
        combined_matches_WB_valid['matched'] = True
        
        # Display the combined DataFrame
        # combined_matches_WB_valid.head()
        
        # Finding all possible matches between bionomia and harvard index (all data)
        index_pairs_HB = generate_match_pairs_Hbio_validated(dfb, dfc, lastName_threshold=0.9, firstName_threshold=0.85)
        # print(len(temp_index))
        print("There are {} possible matches found based on the given criteria".format(len(index_pairs_HB)))
        
        # Print out the precentage
        print('There are '+ str(len(index_pairs_HB)) +' Bionomia records in Harvard Index, which is ' + str(len(index_pairs_HB)/len(b)*100) +'%')
        print('There are '+ str(len(index_pairs_HB)) +' Harvard Index records in Bionomia, which is ' + str(len(index_pairs_HB)/len(c)*100) +'%')
        
        # Merge the matches with the original data
        matched_dfb = dfb.loc[index_pairs_HB.get_level_values('harvardIndex')].reset_index()
        matched_dfc = dfc.loc[index_pairs_HB.get_level_values('bioID')].reset_index()
        
        # Combine the matched DataFrames side by side
        combined_matches_HB = pd.concat([matched_dfb, matched_dfc], axis=1)
        
        # Optionally add a label to identify matched rows
        combined_matches_HB['matched'] = True
        
        # Merge DataFrames based on 'wikiID' to find the intersection
        intersection_true_matches = pd.merge(true_matches_WH.to_frame(index=False), true_matches_WB.to_frame(index=False), on='wikiID')
        
        # Merge DataFrames based on 'wikiID' to find the intersection
        intersection_possible_matches_valid = pd.merge(index_pairs_WH_valid.to_frame(index=False), index_pairs_WB_valid.to_frame(index=False), on='wikiID')
        # print(intersection_possible_matches_valid)
        
        # Find the set difference wiki-HI and wiki-bio
        difference_WH_possible_valid = index_pairs_WH_valid.difference(set(intersection_possible_matches_valid[['wikiID', 'harvardIndex']].apply(tuple, axis=1)))
        difference_WB_possible_valid = index_pairs_WB_valid.difference(set(intersection_possible_matches_valid[['wikiID', 'bioID']].apply(tuple, axis=1)))
        
        # Convert intersection_possible_matches_valid to a set of tuples containing harvardIndex and bioID
        set_valid = set(intersection_possible_matches_valid[['harvardIndex', 'bioID']].apply(tuple, axis=1))
        # Convert index_pairs_HB to a set of tuples containing harvardIndex and bioID
        set_HB = set(index_pairs_HB.to_frame(index=False).apply(tuple, axis=1))
        # Find the difference between the two sets
        difference_set = set_HB - set_valid
        # Convert the resulting set back to a MultiIndex object
        difference_HB_possible_valid = pd.MultiIndex.from_tuples(difference_set, names=['harvardIndex', 'bioID'])
        
        total_match_num = len(intersection_possible_matches_valid) + len(difference_WH_possible_valid) + len(difference_WB_possible_valid) + len(difference_HB_possible_valid)
        
        # Convert Series and Index to sets and find the difference in wiki
        not_matched_set_wiki = set(dfa.index) - (set(index_pairs_WH_valid.get_level_values('wikiID')) | set(index_pairs_WB_valid.get_level_values('wikiID')))
        # Convert the result back to a list or Index if needed in wiki
        not_matched_list_wiki = list(not_matched_set_wiki)
        
        # Convert Series and Index to sets and find the difference in HI
        not_matched_set_HI = set(dfb.index) - (set(index_pairs_WH_valid.get_level_values('harvardIndex')) | set(index_pairs_HB.get_level_values('harvardIndex')))
        # Convert the result back to a list or Index if needed in wiki
        not_matched_list_HI = list(not_matched_set_HI)
        
        # Convert Series and Index to sets and find the difference in Bio
        not_matched_set_bio = set(dfc.index) - (set(index_pairs_WB_valid.get_level_values('bioID')) | set(index_pairs_HB.get_level_values('bioID')))
        # Convert the result back to a list or Index if needed in wiki
        not_matched_list_bio = list(not_matched_set_bio)
        
        total_num_collector = total_match_num + len(not_matched_list_wiki) + len(not_matched_list_HI) + len(not_matched_list_bio)
        
        # Display the resulting figures
        print("Matched records:")
        print("There are {} exact matches found across Wikidata, HI and Bionomia".format(len(intersection_true_matches)))
        print("There are {} possible matches (including all exact matches) found across Wikidata, HI and Bionomia".format(len(intersection_possible_matches_valid)))
        
        print("There are {} more possible matches found between Wikidata and HI".format(len(difference_WH_possible_valid)))
        print("There are {} more possible matches found based across Wikidata and Bionomia".format(len(difference_WB_possible_valid)))
        print("There are {} more possible matches found based across HI and Bionomia".format(len(difference_HB_possible_valid)))
        
        print("\nThere are {} records could be merged across all datasets".format(total_match_num))
        
        print('\n\nNot-matched records:')
        print('There are {} not-matched records in wiki'.format(len(not_matched_list_wiki)))
        print('There are {} not-matched records in HI'.format(len(not_matched_list_HI)))
        print('There are {} not-matched records in Bio'.format(len(not_matched_list_bio)))
        print('\nTotal number of collector records after aggregation is: ' + str(total_num_collector))
        
        # Convert MultiIndex objects to DataFrame
        difference_WH_possible_valid = difference_WH_possible_valid.to_frame(index=False)
        difference_WB_possible_valid = difference_WB_possible_valid.to_frame(index=False)
        difference_HB_possible_valid = difference_HB_possible_valid.to_frame(index=False)
        
        # Merging intersection_true_matches with a, b, and c
        merged_df = intersection_possible_matches_valid.merge(a, left_on='wikiID', right_on='wikiID', suffixes=('', '_w'))
        merged_df = merged_df.merge(b, left_on='harvardIndex', right_on='harvardIndex', suffixes=('', '_h'))
        merged_df = merged_df.merge(c, left_on='bioID', right_on='bioID', suffixes=('', '_b'))
        
        # Adding suffixes manually to avoid conflict in case of multiple merges
        for col in merged_df.columns:
            if col in a.columns and col not in ['wikiID']:
                merged_df.rename(columns={col: col + '_w'}, inplace=True)
            elif col in b.columns and col not in ['harvardIndex']:
                merged_df.rename(columns={col: col + '_h'}, inplace=True)
            elif col in c.columns and col not in ['bioID']:
                merged_df.rename(columns={col: col + '_b'}, inplace=True)
        
        # Merging difference_WH_possible_valid with a and b
        wh_merged = difference_WH_possible_valid.merge(a, on='wikiID', suffixes=('', '_w'))
        wh_merged = wh_merged.merge(b, on='harvardIndex', suffixes=('', '_h'))
        
        # Adding suffixes manually to avoid conflict in case of multiple merges
        for col in wh_merged.columns:
            if col in a.columns and col not in ['wikiID']:
                wh_merged.rename(columns={col: col + '_w'}, inplace=True)
            elif col in b.columns and col not in ['harvardIndex']:
                wh_merged.rename(columns={col: col + '_h'}, inplace=True)
        
        # Merging difference_WB_possible_valid with a and c
        wb_merged = difference_WB_possible_valid.merge(a, on='wikiID', suffixes=('', '_w'))
        wb_merged = wb_merged.merge(c, on='bioID', suffixes=('', '_b'))
        
        # Adding suffixes manually to avoid conflict in case of multiple merges
        for col in wb_merged.columns:
            if col in a.columns and col not in ['wikiID']:
                wb_merged.rename(columns={col: col + '_w'}, inplace=True)
            elif col in c.columns and col not in ['bioID']:
                wb_merged.rename(columns={col: col + '_b'}, inplace=True)
        
        # Merging difference_HB_possible_valid with b and c
        hb_merged = difference_HB_possible_valid.merge(b, on='harvardIndex', suffixes=('', '_h'))
        hb_merged = hb_merged.merge(c, on='bioID', suffixes=('', '_b'))
        
        # Adding suffixes manually to avoid conflict in case of multiple merges
        for col in hb_merged.columns:
            if col in b.columns and col not in ['harvardIndex']:
                hb_merged.rename(columns={col: col + '_h'}, inplace=True)
            elif col in c.columns and col not in ['bioID']:
                hb_merged.rename(columns={col: col + '_b'}, inplace=True)
        
        # Ensure unique columns across all dataframes by renaming columns
        def ensure_unique_columns(df, suffix):
            cols = df.columns.tolist()
            unique_cols = []
            for col in cols:
                if col not in unique_cols:
                    unique_cols.append(col)
                else:
                    unique_cols.append(col + suffix)
            df.columns = unique_cols
            return df
        
        merged_df = ensure_unique_columns(merged_df, '_merged')
        wh_merged = ensure_unique_columns(wh_merged, '_wh')
        wb_merged = ensure_unique_columns(wb_merged, '_wb')
        hb_merged = ensure_unique_columns(hb_merged, '_hb')
        
        # Align columns by filling missing columns with NaN
        all_columns = set(merged_df.columns).union(set(wh_merged.columns), set(wb_merged.columns), set(hb_merged.columns))
        merged_df = merged_df.reindex(columns=all_columns)
        wh_merged = wh_merged.reindex(columns=all_columns)
        wb_merged = wb_merged.reindex(columns=all_columns)
        hb_merged = hb_merged.reindex(columns=all_columns)
        
        # Reset the index of each dataframe before concatenating
        merged_df = merged_df.reset_index(drop=True)
        wh_merged = wh_merged.reset_index(drop=True)
        wb_merged = wb_merged.reset_index(drop=True)
        hb_merged = hb_merged.reset_index(drop=True)
        
        # Combine all merged dataframes
        final_merged_df = pd.concat([merged_df, wh_merged, wb_merged, hb_merged], ignore_index=True)
        final_merged_df.to_csv(self.output().path, index=False)
        

    def output(self): 
        return luigi.LocalTarget(OUTPUT_DIR / 'collectors.csv')    
    
    
if __name__ == "__main__":
    # luigi.build([ProcessSpecimenTask(image_id='011244568', force=True)], local_scheduler=True)
    luigi.build([InstitutionsTask(force=True)], local_scheduler=True)