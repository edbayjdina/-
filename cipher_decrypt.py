from vigenere import decrypt_vigenere

encodings = ['utf-8', 'windows-1251', 'latin-1', 'cp866', 'koi8-r']

encrypted_text = None
for enc in encodings:
    try:
        with open('cipher.txt', 'r', encoding=enc) as f:
            encrypted_text = f.read()
        print(f"Файл успешно прочитан в кодировке: {enc}")
        break
    except UnicodeDecodeError:
        continue

if encrypted_text is None:
    print("Не удалось прочитать файл ни в одной кодировке")
    exit()

print("Первые 200 символов текста:")
print(encrypted_text[:200])
print()

common_keys = ["secret", "lemon", "key", "crypto", "code", "password", "python", "cipher", "caesar", "vigenere", "brilliant", "cryptography"]

found = False
for key in common_keys:
    decrypted = decrypt_vigenere(encrypted_text, key)
    if "the" in decrypted.lower() and "and" in decrypted.lower():
        print(f"Найден ключ: {key}")
        with open('cracked.txt', 'w', encoding='utf-8') as f:
            f.write(decrypted)
        found = True
        break

if not found:
    print("Ключ не найден, пробуем secret")
    decrypted = decrypt_vigenere(encrypted_text, "secret")
    with open('cracked.txt', 'w', encoding='utf-8') as f:
        f.write(decrypted)

print("Готово! Файл cracked.txt создан")