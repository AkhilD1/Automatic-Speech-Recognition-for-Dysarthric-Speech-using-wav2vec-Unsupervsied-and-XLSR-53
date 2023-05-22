import csv

# Define input file name and number of rows per output file
input_file = '/home/015910674/libri_speech_project/dataset/LibriSpeech/train-clean-100/tsv_files/train.tsv'
rows_per_file = 1500

# Open input file and create output files
with open(input_file, 'r') as f_in:
    reader = csv.reader(f_in, delimiter='\t')
    header = next(reader)  # assume first row is header
    file_number = 1
    output_file = f'train_{file_number}.tsv'
    f_out = open(output_file, 'w', newline='')
    writer = csv.writer(f_out, delimiter='\t')
    writer.writerow(header)

    # Iterate over input rows and write to output files
    row_count = 0
    for row in reader:
        writer.writerow(row)
        row_count += 1
        if row_count == rows_per_file:
            f_out.close()
            file_number += 1
            output_file = f'train_{file_number}.tsv'
            f_out = open(output_file, 'w', newline='')
            writer = csv.writer(f_out, delimiter='\t')
            writer.writerow(header)
            row_count = 0

    # Close final output file
    f_out.close()




