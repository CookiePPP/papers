# papers

---

[PnG BERT: Augmented BERT on Phonemes and Graphemes for Neural TTS](https://arxiv.org/pdf/2103.15060.pdf)

The researchers train a BERT model on Phoneme+[SEP]+Text inputs with the standard masked language modelling objective.
They use both word-position and token-position embeddings, along with the usual 2 BERT embeddings for segment A and B.
The position embeddings are sinusoidal and a linear projection is used so the model can seperate each type of embedding in the latent space.
The PnG BERT model is trained on a plain text dataset taken from wikipedia. Their proprietary system is used to convert text into phonemes. Since there is no ground truth data, only predictions taken from their g2p model, this technique might work poorly with accented speakers or multispeaker datasets.
They find that applying the [mask] token to entire words instead of individual tokens improves performance in downstream tasks.

When using the PnG BERT model for TTS, they only use the latents from the phoneme side of the model's output.
Their TTS model is trained on 240 hours of data from 31 speakers.

MOS: The baseline is slightly worse than ground truth
MOS: The PnGBERT augmented solution is equal to ground truth

---

