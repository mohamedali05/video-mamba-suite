
'''
import numpy as np

filename = 'V007.npy'
try:
    feats = np.load(filename, allow_pickle=True)
    print("File loaded successfully.")
except Exception as e:
    print(f"Error loading file {filename}: {e}")

'''

import numpy as np

filename = 'V006.npy'

# Read the raw data
try:
    with open(filename, 'rb') as f:
        raw_data = f.read()
        print(type(raw_data))
    print("Raw data loaded successfully.")
except Exception as e:
    print(f"Error loading raw data from file {filename}: {e}")

# Print the first few bytes to understand the structure
print("First 100 bytes of raw data:", raw_data[:100])

# Attempt to interpret the raw data as a numpy array
try:
    feats = np.frombuffer(raw_data, dtype=np.float32)
    print("Interpreted raw data as numpy array successfully.")
    print("Array shape:", feats.shape)
except Exception as e:
    print(f"Error interpreting raw data as numpy array: {e}")
