import csv
from operator import attrgetter


def read_data_from_csv(filpath):
    training_data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)

        for row in reader:
           # print(row)
            attributes, label = row[:-1], row[-1]
            #print(attributes,label)
            training_data.append((attributes, label))
    print(training_data)
    return training_data

def find_s_algorithm(training_data):
    hypothesis = None
    for example, label in training_data:
        if label == 'Yes':
            if hypothesis is None:
                hypothesis = example[:]
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i] != example[i]:
                        hypothesis[i] = '?'
    return hypothesis

if __name__ == "__main__":
    file_path = "trainingdata.csv"
    training_data = read_data_from_csv(file_path)
    hypothesis = find_s_algorithm(training_data)
    print("Most specific hypothesis:", hypothesis)
