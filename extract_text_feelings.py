import sys
import csv
sys.path.append('../../')


def extract_data():

    data_pathname = "C:/Users/Jiali/PycharmProjects/AROB_En/data/"
    filename = ['data_Boy_Fr', 'data_Comedy_Fr', 'data_Girl_Fr', 'data_Moe_Fr', 'data_Youth_Fr']
    files_nb = len(filename)
    for file_index in range(files_nb):
        with open('C:/Users/Jiali/PycharmProjects/AROB_En/csv_file/' + filename[file_index] + '.csv', 'r',
                  encoding='utf-8') as csv_file:
            content = csv.reader(csv_file, delimiter=';')
            next(content)
            text_file = open(data_pathname + filename[file_index] + "_Text.txt", "w+")
            feeling_file = open(data_pathname + filename[file_index] + "_Feeling.txt", "w+")
            for row in content:
                if row[6] != '' and row[5] != '':
                    text_file.write(row[6].strip() + "\n")
                    feeling_file.write(row[5].strip() + "\n")
                if row[7] != '' and row[5] != '':
                    text_file.write(row[7].strip() + "\n")
                    feeling_file.write(row[5].strip() + "\n" )
    return 0


extract_data()
