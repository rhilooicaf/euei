from transformers import T5Tokenizer

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Now you can use the tokenizer to encode and decode text!
text = "Hello, this is example."
encoded_text = tokenizer.encode(text, return_tensors='pt')
decoded_text = tokenizer.decode(encoded_text[0])

print(f"Encoded text: {encoded_text}")
print(f"Decoded text: {decoded_text}")
