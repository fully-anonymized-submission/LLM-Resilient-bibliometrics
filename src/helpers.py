import os


def write_log(old_sentence, new_sentence, line, start_positions, end_positions, type, log_file):
    # Open log file, write the following lines:
    # 1. old sentence, 2. - for positions that were not changed, * for positions that were changed
    # 3. line and position of the change, 4. type of change, 5. new sentence
    # 6. 3 empty lines
    # Close log file
    # check whether log file exists, otherwise create it
    if not os.path.exists(log_file):
        open(log_file, 'w').close()
    log_file = open(log_file, 'a', encoding='utf-8')
    log_file.write(old_sentence + '\n')
    # start and end positions are lists of integers indicating the positions of the words that were changed
    # in the sentence
    for i in range(len(old_sentence)):
        # if i is between two start and end positions, then put a * in the log file. E..g between start_positions[1] and end_positions[1]
        for k in range(len(start_positions)):
            if i >= start_positions[k] and i < end_positions[k]:
                log_file.write('*')
                break
        else:
            log_file.write('-')
    log_file.write('\n')
    log_file.write('Line: ' + str(line) + ', start: ' + str(start_positions) + ', end: ' + str(end_positions) + ', type: ' + type + '\n')
    log_file.write(new_sentence + '\n\n\n\n')
    log_file.close()
    
################## NOT IN USE CURRENTLY, KEEPS THE LATEST VERSION OF A PAPER WHEN USING A DATAFRAME ##################
def retain_latest_version(df):
    number_of_papers = len(df)
    df['version'] = df['paper'].apply(lambda x: re.findall(r'v(\d+)', x))
    # Now change paper to only include the identifier
    df['paper'] = df['paper'].apply(lambda x: re.findall(r'(\d+\.\d+)', x)[0])

    # find the paper ids where we have multiple versions
    duplicate_ids = df['paper'].value_counts().index[df['paper'].value_counts() > 1]

    # Now we can group by paper and retain the latest version
    df = df.sort_values('version').groupby('paper').tail(1)
    # Now we can drop the version column
    df = df.drop('version', axis=1)

    print(f'From {number_of_papers} papers, we retained {len(df)} papers')
    return df, duplicate_ids