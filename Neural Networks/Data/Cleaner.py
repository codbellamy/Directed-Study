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

        # While loop sometimes does not catch the last entry
        if row == [''] or row == []:
            break

        for column in ACTIVE_COLUMNS:
            # Omitting first column since we don't care about the time elapsed since first capture
            current_column += 1

            # Skip inactive columns
            if column != current_column:
                continue
            else:
                # Class is not a float and time is irrelevant to our testing
                if data[column][0] in ['"Time"','"Class"']:
                    data[column].append(row[column].strip('\n') + ',')
                else:
                    data[column].append(float(row[column]))
        # CSV formatting
        row = f.readline().split(',')
    
    # Note system time taken to read the file
    print("Data read: ", time.time()-start_time)


# Shift and scale data to range from 0-1
for column in range(len(data)):

    # Temporarily remove the headers to perform type casting
    header = data[column].pop(0)
    print("Shifting data for: ", header)

    # Ignore our time and class again
    if header in ['"Time"','"Class"']:
        continue

    # Determine how far to shift data so all values are positive
    shift = -min(data[column])

    # Determine how much to scale the values based on the largest and smallest value
    diff = max(data[column]) + shift

    # Perform shift, (value + shift) / scaling_factor
    for row in range(len(data[column])):
        data[column][row] = str((data[column][row] + shift) / diff) + ','

# Output shifted data
with open('Neural Networks/Data/output.csv','w') as f:

    # Replace headers
    for header in HEADERS:
        if header == '"Time"':
            continue
        f.write(header + ",")
    f.write('\n')

    print('Writing to file...')
    start_time = time.time()

    # Writing to CSV requires a transpose, we need to write row-wise rather than column-wise
    for row in range(len(data[1])):

        # Console output to keep track of progress
        if row % 10000 == 0:
            print('Completed', row, 'rows.', time.time()-start_time)
        
        # A list comprised of each row
        transpose = [column[row] for column in data if data.index(column) != 0]

        # We already added commas in the previous for loop, simply write the entry and a comma
        for element in transpose:
            f.write(str(element))
        
        # End of row, next row
        f.write('\n')
    
    print("Done.", time.time()-start_time)