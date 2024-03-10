import sys
import zlib
import random
import string

def randomStringGenerator(length):
    return ''.join(random.choices(string.ascii_letters, k=length))

input_str = randomStringGenerator(1000)

# Checking size of input string
str_size = sys.getsizeof(input_str)
print("Size of original text", str_size)

# Compressing text
compressed = zlib.compress(input_str.encode())

# Checking size of text after compression
csize = sys.getsizeof(compressed)
print("Size of compressed text", csize)

# Decompressing text
decompressed = zlib.decompress(compressed).decode()

# Checking size of text after decompression
dsize = sys.getsizeof(decompressed)

# Check it is lossless compression
assert dsize == str_size
print("Decompressed text has the same size {}, so it is lossless compression.".format(dsize))

# Calculate compression ratio
print("Compression ratio is {:.4f}".format(str_size / csize))