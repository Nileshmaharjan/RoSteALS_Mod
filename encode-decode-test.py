from reedsolo import RSCodec


def encode_message(secret_message, additional_bits):
    # Convert secret message to bits
    secret_bits = ''.join(format(ord(char), '08b') for char in secret_message)

    # Reed-Solomon encoding with a fixed number of error-correction symbols
    rs = RSCodec(7)  # Using 40 error-correction symbols to always produce a 200-bit output
    encoded_data = rs.encode(int(secret_bits, 2).to_bytes((len(secret_bits) + 7) // 8, byteorder='big'))

    # Convert to binary data and ensure it is exactly 200 bits
    binary_data = ''.join(format(value, '08b') for value in encoded_data)

    binary_data += '0' * additional_bits
    print('here')

    return binary_data


def decode_message(binary_data, additional_bits):


    # Remove additional bits added during encoding
    binary_data = binary_data[:-additional_bits]

    # Convert binary data back to bytes
    encoded_bytes = bytes([int(binary_data[i:i + 8], 2) for i in range(0, len(binary_data), 8)])

    # Reed-Solomon decoding with the same fixed number of error-correction symbols
    rs = RSCodec(7)
    try:
        decoded_data = rs.decode(list(encoded_bytes))[0].decode('utf-8')
        return decoded_data
    except:
        return "Error: Unable to decode. Data may be corrupted."


# Example usage
secret_message = "changwon_national"  # Example with a shorter message\

if len(secret_message) == 1:
    additional_bits = 200 - (8 * 8)

else:
    additional_bits = 200 - (8 * (8 + (len(secret_message) - 1)))

encoded_data = encode_message(secret_message, additional_bits)
decoded_message = decode_message(encoded_data, additional_bits)

print("Original Message:", secret_message)
print("Encoded Data:", encoded_data)
print("Decoded Message:", decoded_message)
