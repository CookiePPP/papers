# papers

---

[PnG BERT: Augmented BERT on Phonemes and Graphemes for Neural TTS](https://arxiv.org/pdf/2103.15060.pdf)

The researchers train a BERT model on Phoneme+[SEP]+Text inputs with the standard masked language modelling objective.
They use both word-position and token-position embeddings, along with the usual 2 BERT embeddings for segment A and B.
The position embeddings are sinusoidal and a linear projection is used so the model can seperate each type of embedding in the latent space.
The PnG BERT model is trained on a plain text dataset taken from wikipedia. Their proprietary system is used to convert text into phonemes. Since there is no ground truth data, only predictions taken from their g2p model, this technique might work poorly with accented speakers or multispeaker datasets if no fine-tuning is used.
They find that applying the [mask] token to entire words instead of individual tokens improves performance in downstream tasks.

When using the PnG BERT model for TTS, they only use the latents from the phoneme side of the model's output.
Their TTS model is trained on 240 hours of data from 31 speakers.

Results
The baseline is slightly worse than ground truth. The PnG-BERT augmented solution is better/equal to ground truth.
Reviewers found that the PnG BERT solution had better prosody and naturalness.

---

[Neural Codec Language Models are
Zero-Shot Text to Speech Synthesizers](https://arxiv.org/pdf/2301.02111.pdf) (VALL-E)

Following the success of large language models, the researchers experiment with large speech models.
They use facebook's "EnCodec" VQVAE model to convert 24Khz audio into descrete audio tokens, then train a BERT-large style model with casual language modelling task on the audio tokens.
The EnCodec model can output {1,2,4,8} tokens per timestep depending on the amount of compression being targeted.
To improve performance, the CLM only predicts the first token for each timestep autoregressively, then a (non-autoregressive) BERT-large predicts the remaining 7 audio tokens in parallel.
The authors use a lot of weight/layer sharing among the 7 parallel BERT-large models, presumably to speed up training or reduce parameter count since the task of each model is almost identical.

To train this architecture, they use the Libri-light 60,000 dataset (where an ASR model transcribes each file).
They use 10-20 second slices to train the autoregressive model.
3 Seconds of ground truth audio tokens are given at the start of sample (so the model can learn the audio conditions and speaker identity for the sample).
They also provide the phoneme transcript at the start of each sample. Different position embeddings are used for the phoneme transcript to allow the model to align the transcript and audio tokens properly.

Results
They find that VALL-E has better WER and speaker similarity than YourTTS (zero-shot model based on VITS), however there is still a sigificant distance from ground truth.
The MOS values suggest VALL-E has very good audio quality and naturalness.
![image](https://user-images.githubusercontent.com/42448678/216848288-2ca87972-9b0f-4aa8-9887-659be660896e.png)
![image](https://user-images.githubusercontent.com/42448678/216848300-1132ccb5-ed40-4848-8dfa-e18b7c208fcc.png)

Thoughts
The speaker similarity could be improved by giving the model more ground truth audio to work with. The WER rate should be ignored in this paper since the model is trained on ASR transcriptions. If the dataset transcripts were created using the same model that was used to evaluate the WER rate then the WER rate may be inflated/incorrect.
