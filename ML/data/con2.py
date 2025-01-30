# Open the input text file and read its contents
input_file = 'head.txt'  # Replace with your input text file name
output_file = 'outputhead.csv'  # Replace with the desired output CSV file name

with open(input_file, 'r') as file:
    data = file.read()

# Replace spaces with commas
data =data.replace(',',':')
data = data.replace(';', ',')

# Write the modified data to the output CSV file
with open(output_file, 'w') as file:
    file.write(data)

print(f"File has been processed and saved as {output_file}")
