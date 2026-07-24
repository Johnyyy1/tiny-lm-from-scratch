data_file = '/Users/jonas/Downloads/training_data.txt'

with open(data_file, 'r', encoding='utf-8') as handle:
    data = [line.rstrip('\n') for line in handle]

split_index = int(len(data) * 0.9)
data_trn = data[:split_index]
data_val = data[split_index:]

# Get the unique characters across the training set
unique_chars = set()

for story in data_trn:
    unique_chars.update(set(story))

unique_chars = ''.join(sorted(unique_chars))

# We use '^' as a special start character in the model
assert '^' not in unique_chars

stop_char = '^'

# Add stop character to each string
data_trn[:] = [s + "^" for s in data_trn]
data_val[:] = [s + "^" for s in data_val]