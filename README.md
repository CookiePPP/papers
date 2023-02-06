# papers

---

## [PnG BERT: Augmented BERT on Phonemes and Graphemes for Neural TTS](https://arxiv.org/pdf/2103.15060.pdf)

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

## [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/pdf/2301.02111.pdf) (VALL-E)

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

Thoughts: The WER rate should be ignored in this paper since the model is trained on ASR transcriptions. If the dataset transcripts were created using the same model that was used to evaluate the WER rate then the WER rate may be inflated/incorrect.

They also evaluated speaker similarity using 5s and 10s ground truth tokens for input and found that speaker similarity improves as more data is used (unsurprisingly).

![image](https://user-images.githubusercontent.com/42448678/217071911-8237fef7-d001-42ca-bbd9-0b3770a9393f.png)

They also mention the model being able to recreate reverberation when it's given in ground truth samples. That's very interesting given models like Tacotron2 and FastSpeech2 struggle with reverberation.

---

## [ResGrad: Residual Denoising Diffusion Probabilistic Models for Text to Speech](https://arxiv.org/pdf/2212.14518.pdf)

The researchers found that the text-to-speech model FastSpeech2 does not produce sharp/clear spectrograms, especially with challenging speakers or large multi-speaker datasets.
In order to fix this problem, they train a diffusion model to learn the offset between FastSpeech2's output and the ground truth data. Effectively using the Diffusion model as a postnet.
They find that it's very effective, as few as 4 sampling steps is enough to improve the MOS from 3.3 to 4.1.
The authors spend a long time talking about how their method is faster than other Diffusion TTS models, however they seem to completely misunderstand or mis-explain WHY it's faster. The model is faster because their model uses ground truth pitch during training and only outputs 1 pitch value during inference. Because of this, their model can only generate 1 version of each audio file.
I threw away FastSpeech2 and trained a normal Diffusion model with ground truth pitch as aux input, and it also produces samples in less than 10 steps, while enjoying the significantly simpler architecture design.

---

## [Regotron: Regularizing the Tacotron2 architecture via monotonic alignment loss](https://arxiv.org/pdf/2204.13437.pdf)

The researchers find that Tacotron2 often learns non-monotonic alignments.
Inorder to fix this, they calculate the mean text-position of each mel-frames alignment vector, and minimize the negative difference between the position of neighbouring frames.
Basically, they add a new loss the will penalize the model if the position in the text goes backwards by any amount between mel frames.
![image](https://user-images.githubusercontent.com/42448678/216850141-5e28c5a7-1c74-473b-aeea-4b2f0e43b670.png)
They find that this improves the number of skipped or repeated words and show a minor increase in MOS.

I like this paper. It's an extremely simple technique that just works and doesn't seem to have any downsides (at least with phoneme input text).

---

## [JETS: Jointly Training FastSpeech2 and HiFi-GAN for End to End Text to Speech](https://arxiv.org/pdf/2203.16852.pdf)

The researchers find that using seperate Alignment, Text-to-Spectrogram and Vocoder models may reduce the quality of text-to-speech samples.
(FastSpeech2 has actually tried this before with "FastSpeech2s", however they reported worse scores in their paper.)

For their experiment, they attach FastSpeech2 to HiFiGAN and replace FastSpeech2's hard alignments with a Guassian upsampling aligner.
They use MAS to compute their alignments and remove the mel-spectrogram loss from FastSpeech2 so there are no spectrograms used in this pipeline.
LJSpeech with default model/data parameter are used for training.
They find that despite achieving worse MCD compared to the normal FastSpeech2+HiFiGAN pipeline, they have better F0, MOS and CER.
![image](https://user-images.githubusercontent.com/42448678/216851211-6f2a50d0-75cb-40ef-8b46-16115c0908e7.png)
The difference is significant, however I can't say how much of the difference comes from the alignment change and how much comes from training end-to-end without using spectrograms.
Vocoders are expensive to train so I don't see this architecture becoming common in research anytime soon, but it's still interesting to see and suggests that end-to-end training may be a way to improve audio quality in the future.

---

## [StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis](https://arxiv.org/pdf/2205.15439.pdf)

The researchers propose a Text-to-Speech architecture that can copy a reference audio files prosody/emotion while being given a new piece of text.

At first glance, this architecture appears to be a parallel version of [Global-Style-Tokens](https://arxiv.org/pdf/1803.09017.pdf)
![image](https://user-images.githubusercontent.com/42448678/216851891-b8dd0886-4fba-4410-9a10-0b78940f5c62.png)

They use Tacotron2's text encoder, and Tacotron2's LSA Attention + LSTM Stack for Alignment during training.
The style encoder is just 4 ResBlocks followed by a timewise average over each channel.
Pitch is extracted/used like normal.
The decoder is 7 ResBlocks using AdaIN normalization
They use 4 ResBlocks for a spectrogram discriminator.
They use 3 BiLSTM's with AdaIN for the duration predictor (wow, that's a weird/interesting design decision).
And they predict the NRG+F0 using GT GSTs and the text.

For some reason they train this model in 2 stages, first they train the Decoder with GT F0 + GT GST, then they freeze most of the model and train the GST, Dur, NRG, F0 predictors

They train on LibriTTS 250 hour dataset with 1151 speakers.

![image](https://user-images.githubusercontent.com/42448678/216852261-00922d4f-ce31-44ff-81c3-2149d137ae0a.png)
![image](https://user-images.githubusercontent.com/42448678/216852309-564eb523-c7a6-43fd-868a-1b2e3a09e9a3.png)

They show very good MOS values for Naturalness and Similarity. I'm definitely skeptical of their conclusions / results.
They claim their Style Encoder was able to extract correct emotion from other speakers when the model was trained on a single speaker dataset, yet I don't see anything in their paper that would explain how this is possible.

![image](https://user-images.githubusercontent.com/42448678/216852386-046a55f2-b3fc-451d-ae93-737cb4ec9b8b.png)

They perform lots of Ablations and show that enforcing hard monotonicity is required for parallel architectures to align well (disappointing but expected).
They also show an extremely large drop in quality when the discriminator is removed, which makes me more interested in their discriminator design. I've tried multiple spectrogram discriminators and while having one is better than none, I've found that there's a lot of room for failure and improvements in disciminator design (e.g: including text encoder information, using 2d convs).
They also show that their use of Instance Norm is essential for their architecture, however Adapative Instance Norm is not specifically required.

TODO: Check out their [github repo](https://github.com/yl4579/StyleTTS) and clean this section up. The paper is very dense with information and there's too much to understand with a quick skim.

---

## [Differentiable Duration Modeling for End-to-End Text-to-Speech](https://arxiv.org/pdf/2203.11049.pdf)

![image](https://user-images.githubusercontent.com/42448678/216853590-fda341a6-60fc-4ffd-bc7a-d988f7b4bf5a.png)

The researchers attempt to solve the issue of parallel models requiring external alignment by using the durations from a duration predictor for training the model.
The image above says everything you need to know.

---

## [PortaSpeech: Portable and High-Quality Generative Text-to-Speech](https://arxiv.org/pdf/2109.15166.pdf)

The researchers experiment with various methods of improving TTS quality, reducing model size and increasing throughput.

They note that hard-alignments may reduce naturalness since in actual speech phonemes blend together and don't have well defined boundardies. To fix this they add a word encoder and predict word-level durations instead of phoneme level, then they train an attention module to expand the word-level alignments to phoneme-level.
![image](https://user-images.githubusercontent.com/42448678/216856237-7981f3c2-1ba1-4d9e-9418-9eadf920a2d3.png)

I absolutely love this idea. You get the robustness of hard alignments and the naturalness of soft alignments at the same time, and it doesn't use almost any additional compute. This idea could also be extended to other prosody based features, or added as an additional step for cascading inference like a better FastSpeech2.

They also experiment with using a unconditional VAE to compress the spectrogram, then a conditional NF to infer the VAE latent. I'm not sure why this method has become common but VITS found success with it so I guess it has some merit. They also have a NF Postnet to produce the final spectrogram, which is typically done because VAEs trained with MSE produce blurry outputs.
The postnet significantly improves audio quality, while the VAE+NF latent modelling significantly improves prosody.

---

## [Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech](https://arxiv.org/pdf/2105.06337.pdf) and [Diff-TTS: A Denoising Diffusion Model for Text-to-Speech](https://arxiv.org/pdf/2104.01409.pdf)

These are the first papers to apply DDPMs to the text-to-spectrogram task.

Grad-TTS uses a UNet architecture while Diff-TTS uses Gated WaveNet blocks.

![image](https://user-images.githubusercontent.com/42448678/216861491-e7a0f231-1760-4964-a3c9-540c11ebc4dd.png)

![image](https://user-images.githubusercontent.com/42448678/216861533-07283dcc-641c-488b-91cf-d7b893834c0a.png)

Both papers report SOTA results, with Diff-TTS getting BETTER THAN GT MOS values. Crazy stuff.

---

## [ON GRANULARITY OF PROSODIC REPRESENTATIONS IN EXPRESSIVE TEXT-TO-SPEECH](https://arxiv.org/pdf/2301.11446.pdf)

![image](https://user-images.githubusercontent.com/42448678/216903283-7626112f-c38d-48aa-b97b-2aeea892300a.png)

The researchers experiment with a word-level VAE for encoding prosody of the text.
They find that phoneme-level prosody is hard to predict from text, while utterance level prosody doesn't contain enough information to improve the decoder's results.
To convert the VAE frame-level latents to word-level, they select the middle mel-frame in each word instead of using an RNN or other seq2vec technique.
They use a pretrained BERT model to predict the VAE's latents for inference and achieve extremely good MOS results.

![image](https://user-images.githubusercontent.com/42448678/216904014-fdc96428-2b6f-4cd0-a834-b7cdefcc3caf.png)

---

## [Guided-TTS 2: A Diffusion Model for High-quality Adaptive Text-to-Speech with Untranscribed Data](https://arxiv.org/pdf/2205.15370.pdf)

In Guided-TTS2 the researchers experiment with training a large diffusion model on unlabelled Librispeech 60,000 hours dataset.
They train the DDPM with speaker embeddings taken from a pretrained speaker encoder.
They train the DDPM without conditioning during some of the samples so they can use Classifier-Free-Guidance during inference to closer match the encoded speaker embedding.
No text is used to train the DDPM.

![image](https://user-images.githubusercontent.com/42448678/217060867-d2342b53-1d23-4699-b53b-be69e711ee53.png)

---

Inference

They use a pretrained phoneme classifier to guide the DDPM towards the text for inference.

To make outputs closer to the reference speaker, they use Classifier-Free-Guidance. They notice that CFG reduces the text accuracy but increases the speaker similarity and settle on CFG scale of 1.0 for their evaluations.

![image](https://user-images.githubusercontent.com/42448678/217061875-3ca8d0f0-10bd-4012-b3f7-e56954e316f8.png)

![image](https://user-images.githubusercontent.com/42448678/217062367-6058bc86-8f3a-4cb1-8d45-facaae4334ae.png)

Unless otherwise specified, they also fine-tune the DDPM model on the reference files. This gives a very large improvement in speaker similarity without affecting text accuracy at lower CFGs. (Yellow vs Blue line in the above images)

---

![image](https://user-images.githubusercontent.com/42448678/217063373-ab7e0f69-59ce-46a8-9584-76bfdb283607.png)

Their results are very impressive, outperforming YourTTS, StyleSpeech and matching or exceeding GT in MOS. They achieve slightly below GT speaker similarity, but still significantly better than the competition. Their CER results might be misleading though since they use a pretrained ASR model to guide their DDPM.

https://ksw0306.github.io/guided-tts2-demo/

Listening to their demo samples, I notice each sample sounds very clear and well paced, however emotion seems to be completely missing. It makes sense given their method, but I am curious if semantic information could be added to their method while still being able to train on Librilight. (maybe Whisper(?) transcribed audio, but with a 'AI Transcript' and 'Human Transcript' label added?)

---

![image](https://user-images.githubusercontent.com/42448678/217069712-2307ca13-1a8c-402c-8c29-0254b7a5763b.png)

Also super interesting, they find that resetting the optimizer improves fine-tuning results significantly.

---
