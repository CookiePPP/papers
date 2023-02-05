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

To train this architecture, they use the Libri-light 60,000 hour dataset (where an ASR model transcribes each file).
They use 10-20 second slices to train the autoregressive model.
3 Seconds of ground truth audio tokens are given at the start of sample (so the model can learn the audio conditions and speaker identity for the sample).
They also provide the phoneme transcript at the start of each sample. Different position embeddings are used for the phoneme transcript to allow the model to align the transcript and audio tokens properly.

Results
They find that VALL-E has better WER and speaker similarity than YourTTS (zero-shot model based on VITS).
The MOS values suggest VALL-E has very good audio quality and naturalness.
![image](https://user-images.githubusercontent.com/42448678/216848288-2ca87972-9b0f-4aa8-9887-659be660896e.png)
![image](https://user-images.githubusercontent.com/42448678/216848300-1132ccb5-ed40-4848-8dfa-e18b7c208fcc.png)

Thoughts
The WER rate should be ignored in this paper since the model is trained on ASR transcriptions. If the dataset transcripts were created using the same model that was used to evaluate the WER rate then the WER rate may be inflated/incorrect.

They also evaluated speaker similarity using 5s and 10s ground truth tokens for input and found that speaker similarity improves as more data is used (unsurprisingly).

They also mention the model being able to recreate reverberation when it's given in ground truth samples. That's very interesting given models like Tacotron2 and FastSpeech2 struggle with reverberation.

---

[ResGrad: Residual Denoising Diffusion Probabilistic Models for Text to Speech](https://arxiv.org/pdf/2212.14518.pdf)

The researchers found that the text-to-speech model FastSpeech2 does not produce sharp/clear spectrograms, especially with challenging speakers or large multi-speaker datasets.
In order to fix this problem, they train a diffusion model to learn the offset between FastSpeech2's output and the ground truth data. Effectively using the Diffusion model as a postnet.
They find that it's very effective, as few as 4 sampling steps is enough to improve the MOS from 3.3 to 4.1.
The authors spend a long time talking about how their method is faster than other Diffusion TTS models, however they seem to completely misunderstand or mis-explain WHY it's faster. The model is faster because their model uses ground truth pitch during training and only outputs 1 pitch value during inference. Because of this, their model can only generate 1 version of each audio file.
I threw away FastSpeech2 and trained a normal Diffusion model with ground truth pitch as aux input, and it also produces samples in less than 10 steps, while enjoying the significantly simpler architecture design.

---

[Regotron: Regularizing the Tacotron2 architecture via monotonic alignment loss](https://arxiv.org/pdf/2204.13437.pdf)

The researchers find that Tacotron2 often learns non-monotonic alignments.
Inorder to fix this, they calculate the mean text-position of each mel-frames alignment vector, and minimize the negative difference between the position of neighbouring frames.
Basically, they add a new loss the will penalize the model if the position in the text goes backwards by any amount between mel frames.
![image](https://user-images.githubusercontent.com/42448678/216850141-5e28c5a7-1c74-473b-aeea-4b2f0e43b670.png)
They find that this improves the number of skipped or repeated words and show a minor increase in MOS.

I like this paper. It's an extremely simple technique that just works and doesn't seem to have any downsides (at least with phoneme input text).
