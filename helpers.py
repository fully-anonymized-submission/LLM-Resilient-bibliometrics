import os


def write_log(old_sentence, new_sentence, line, start_positions, end_positions, type, log_file):
    """ Write a log file with the changes made to the sentences in the corpus.

        Parameters
            old_sentence (str): the original sentence
            new_sentence (str): the sentence after the changes
            line (int): the line number in the file
            start_positions (list): the starting positions of the words that were changed
            end_positions (list): the ending positions of the words that were changed
            type (str): the type of change that was made
            log_file (str): the name of the log file

        Returns
            None

    """
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
