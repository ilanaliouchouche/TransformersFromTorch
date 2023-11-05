import sys
import os
from transformers import GPT2Tokenizer
import torch

# Add the parent directory to sys.path to access gpt_classes
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from GPT import gpt_classes

# Define constants
EMBED_DIM = 768
BATCH_SIZE = 1
VOCAB = 50257
POS = 1024
HEADS = 12

'''
Load and tokenize the input text:
- Utilize the GPT2 tokenizer to convert text into token IDs.
'''
model = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model)
text = "This is a file for unit tests"
inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
SEQ_LEN = inputs.input_ids.shape[1]

'''
Define expected tensor attributes for validation:
- SIZE: Expected shape of the tensor.
- DTYPE: Expected data type of the tensor.
- DEVICE: Expected device of the tensor.
'''
SIZE = torch.Size([BATCH_SIZE, SEQ_LEN, EMBED_DIM])
DTYPE = torch.float32
DEVICE = inputs.input_ids.device

'''
Instantiate and test the Embeddings module:
- The Embeddings module converts token IDs to embeddings.
- The output tensor is validated for shape, data type, and device.
'''
token_emb = gpt_classes.Embeddings(VOCAB, EMBED_DIM, POS)
inputs_embedd = token_emb(inputs.input_ids)
assert inputs_embedd.size() == SIZE, "Shape error with gpt_classes.Embeddings"
assert inputs_embedd.dtype == DTYPE, "dtype error with gpt_classes.Embeddings"
assert inputs_embedd.device == DEVICE, "Device error with gpt_classes.Embeddings"

'''
Instantiate and test the MultiHeadAttention module:
- The MultiHeadAttention module applies the attention mechanism to the input embeddings.
- The output tensor is validated for shape, data type, and device.
'''
mlt_attn = gpt_classes.MultiHeadAttention(EMBED_DIM, HEADS)
after_attn = mlt_attn(inputs_embedd)
assert after_attn.size() == SIZE, "Shape error with gpt_classes.MultiHeadAttention"
assert after_attn.dtype == DTYPE, "dtype error with gpt_classes.MultiHeadAttention"
assert after_attn.device == DEVICE, "Device error with gpt_classes.MultiHeadAttention"

'''
Instantiate and test the FeedForward module:
- The FeedForward module applies a feed-forward neural network to the output of the attention mechanism.
- The output tensor is validated for shape, data type, and device.
'''
ff = gpt_classes.FeedForward(EMBED_DIM, VOCAB)
after_ff = ff(after_attn)
assert after_ff.size() == SIZE, "Shape error with gpt_classes.FeedForward"
assert after_ff.dtype == DTYPE, "dtype error with gpt_classes.FeedForward"
assert after_ff.device == DEVICE, "Device error with gpt_classes.FeedForward"

'''
Instantiate and test the Decoder module:
- The Decoder module passes the input through a decoder layer.
- The output tensor is validated for shape, data type, and device.
'''
encode = gpt_classes.Decoder(EMBED_DIM, VOCAB, HEADS)
out = encode(after_ff)
assert out.size() == SIZE, "Shape error with gpt_classes.Decoder"
assert out.dtype == DTYPE, "dtype error with gpt_classes.Decoder"
assert out.device == DEVICE, "Device error with gpt_classes.Decoder"
