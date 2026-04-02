def encrypt_vigenere(plaintext: str, keyword: str) -> str:
    ciphertext = ""
    keyword_index = 0
    for char in plaintext:
        if char.isalpha():
            keyword_char = keyword[keyword_index % len(keyword)]
            shift = ord(keyword_char.lower()) - ord('a')
            start = ord('A') if char.isupper() else ord('a')
            encrypted_char = chr((ord(char) - start + shift) % 26 + start)
            ciphertext += encrypted_char
            keyword_index += 1
        else:
            ciphertext += char
    return ciphertext


def decrypt_vigenere(ciphertext: str, keyword: str) -> str:
    plaintext = ""
    keyword_index = 0
    for char in ciphertext:
        if char.isalpha():
            keyword_char = keyword[keyword_index % len(keyword)]
            shift = ord(keyword_char.lower()) - ord('a')
            start = ord('A') if char.isupper() else ord('a')
            decrypted_char = chr((ord(char) - start - shift) % 26 + start)
            plaintext += decrypted_char
            keyword_index += 1
        else:
            plaintext += char
    return plaintext