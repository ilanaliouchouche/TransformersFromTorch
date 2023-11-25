from t5_classes import *
from torch import nn


class MyT5(nn.Module):
    def __init__(self, vocab_size, position_size, embedding_dim, intermediate_dim, num_attention_heads, num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, embedding_dim, position_size)
        self.encoders = nn.ModuleList([Encoder(embedding_dim, intermediate_dim, num_attention_heads) for _ in range(num_encoder_layers)])
        self.decoders = nn.ModuleList([Decoder(embedding_dim, intermediate_dim, num_attention_heads) for _ in range(num_decoder_layers)])

    def forward(self, input_ids, output_ids):
        encoder_input = self.embeddings(input_ids)
        encoder_output = encoder_input
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output)
        decoder_input = self.embeddings(output_ids)
        decoder_output = decoder_input
        for decoder in self.decoders:
            decoder_output = decoder(decoder_output, encoder_output)
        return decoder_output
