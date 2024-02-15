import argparse
import os
import json
import re

# Create the parser
parser = argparse.ArgumentParser(description="Usage: python3 make_answer_key.py --input_file INPUT_FILE --output_file OUTPUT_FILE")

# Add arguments
parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', type=str)

args = parser.parse_args()

output_file = open(args.output_file, 'w')

#text --> query | answer_key
#pattern = re.compile(r'^([^ ]+)')
#pattern2 = re.compile(r'^([^ ]+)')
pattern = re.compile(r'(cpd\d+)')
pattern2 = re.compile(r'(cpd\d+)')
test_filename = args.input_file
super_index_list = []
first_index_list = []
output_file.write("#query_keyword\tanswer_key\n")
with open(test_filename, 'r', encoding='utf-8') as file:
    for line in file:
        # Strip leading and trailing whitespace from the line
        stripped_line = line.strip()
        # Skip the iteration if the line is empty after stripping
        if not stripped_line:
            continue
        line = stripped_line
        #print(line)
        name = pattern.search(line)
        if name:
            query = name.group()
            answer_key = pattern2.search(line)
            #print("#", query, answer_key.group())
            #answer_key = name.group()
            if answer_key:
                output_file.write(query + "\t" + answer_key.group() + "\n")

output_file.close()
