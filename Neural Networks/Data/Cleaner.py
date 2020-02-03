# Data formatting tool to clean up our test data for reading credit cards
# Very rough and dirty, just to get the job done. Could be cleaner, but it works
# By Cody Bellamy

import time, numpy as np

# Columns V1-V28 contain capture data, Amount and Class follow.
# Amount = the dollar amount the transaction was for
# Class = Binary response on weather the transaction was fraudulent(1) or not(0)
ACTIVE_COLUMNS = [i for i in range(1,31)]
data = []

with open('Neural Networks/Data/creditcard.csv','r') as f:
    HEADERS = f.readline().split(',')
    data = [[header.strip('\n')] for header in HEADERS]
    row = f.readline().strip('\n').split(',')
    start_time = time.time()
    print("Reading data...")

    # Cycle through every row of the data
    while row != [''] or row != []:
        current_column = 0

        if row == [''] or row == []:
            break

        for column in ACTIVE_COLUMNS:
            # Omitting first column since we don't care about the time elapsed since first capture
            current_column += 1

            # Skip inactive columns
            if column != current_column:
                continue
            else:
                if data[column][0] in ['"Time"','"Class"']:
                    data[column].append(row[column].strip('\n') + ',')
                else:
                    data[column].append(float(row[column]))

        row = f.readline().split(',')
    print("Data read: ", time.time()-start_time)



for column in range(len(data)):
    header = data[column].pop(0)
    print("Shifting data for: ", header)
    if header in ['"Time"','"Class"']:
        continue
    shift = -min(data[column])
    diff = max(data[column]) + shift
    for row in range(len(data[column])):
        data[column][row] = str((data[column][row] + shift) / diff) + ','

with open('Neural Networks/Data/output.csv','w') as f:

    for header in HEADERS:
        if header == '"Time"':
            continue
        f.write(header + ",")
    f.write('\n')

    print('Writing to file...')
    start_time = time.time()

    for row in range(len(data[1])):
        if row % 10000 == 0:
            print('Completed', row, 'rows.', time.time()-start_time)
        transpose = [column[row] for column in data if data.index(column) != 0]
        for element in transpose:
            f.write(str(element))
        f.write('\n')
    
    print("Done.", time.time()-start_time)