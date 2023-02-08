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

## [Mixed-Phoneme BERT: Improving BERT with Mixed Phoneme and Sup-Phoneme Representations for Text to Speech](https://arxiv.org/pdf/2203.17190.pdf)

This paper continues the research of PnG BERT in Prosody/Pronunciation prediction using Large Language Models.

They experiment with training a BERT model with only phonemes. They use BPE to create a large vocabulary and train with both sub-phonemes and phonemes as can be seen in the image below.

![image](https://user-images.githubusercontent.com/42448678/217095512-4de69ca0-1bac-4e18-ae18-9beee818de16.png)

---

They show that

![image](https://user-images.githubusercontent.com/42448678/217095685-071085af-20bd-4674-9f79-65230d262df4.png)

removing the text side of the model/input significantly improves performance without any decrease in MOS.

---

They show that

![image](https://user-images.githubusercontent.com/42448678/217095911-a70badb5-6c95-45fe-be0a-3c8664d62eb3.png)

the sub-phoneme vocabulary is required for the LLM to learn the required information.

The results are great and show that their method is recommended if have phonemes available for inference or a dataset with a common nationality/accent.

---

> The TTS front-end normalizes the sentences and converts them into phoneme sequences.

Since they use a generic g2p algorithm for inference, this approach may not work as well with strong accented speakers or a very diverse multispeaker dataset.

---

## [NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality](https://arxiv.org/pdf/2205.04421.pdf) (Clickbate title 👏)

In this paper the researchers show that Mixed-Phoneme BERT and Parallel Tacotron2 results are reproducible. The researchers also propose a new architecture with "Bidirectional Prior/Posterior" and "VAE with Memory" however in my opinion they do not show how these techniques compare to existing methods properly.

<details>
  <summary>Click for Details</summary>

In this paper they merge;
- Mixed-Phoneme BERT
- Parallel tacotron 2: A non-autoregressive neural tts model with differentiable duration modeling
- VITS

and use a slightly customized NF prior.

![image](https://user-images.githubusercontent.com/42448678/217080581-4e5e2bca-24ae-4d39-bce4-0febfc059ffc.png)

![image](https://user-images.githubusercontent.com/42448678/217089226-4ada8ad3-05ae-48d0-90f4-fa28a25c93ae.png)


They train their model on LJSpeech and get almost perfect MOS values.
They do not however show GT MEL + HiFiGAN, so as far as I know, Grad-TTS/Glow-TTS is actually better than NaturalSpeech and they're just being held back by the Vocoder. Rather frustrating that they spend so much of the paper on their sampling method, but don't ever perform an apples-to-apples comparison to show if it's actually any better.

---

![image](https://user-images.githubusercontent.com/42448678/217080976-dba517bd-9532-4e9c-b9dc-ad8df0d48505.png)

They show that each change has a minor positive impact on the MOS. Phoneme Pretraining and Differentiable Durator come from Mixed-Phoneme BERT and Parallel tacotron 2 respectively.

---

![image](https://user-images.githubusercontent.com/42448678/217083087-ace2668c-9694-46e3-875d-7996f2fd53b2.png)

They evaluate FastSpeech2 + HifiGANV1 and show how train/inference mismatch at each stage results in lower in MOS scores.
They use GT PT HiFiGAN so the Mel Decoder result should be ignored.

The Vocoder result is interesting. Lots of vocoders are better than HiFiGAN and HiFiGAN appears to be almost perfect... so maybe we don't have to waste lots of compute training E2E models in the future after all?

The Phoneme Encoder result is really nice to see. We've now got 9(?) examples of LLM based text encodering massively improving results, so it's clearly one of the next big things to hit open-source.

</details>
  
---

## [Revisiting Over-Smoothness in Text to Speech](https://arxiv.org/pdf/2202.13066.pdf)

In this paper the researchers evaluate various sampling methods side-by-side

![image](https://user-images.githubusercontent.com/42448678/217098977-4c1ec41e-8a32-4b72-bd73-5e52cf1c6401.png)

![image](https://user-images.githubusercontent.com/42448678/217099014-dd4a2f35-a05e-44bd-93d4-d9a9c0461bd1.png)

![image](https://user-images.githubusercontent.com/42448678/217099034-a8e132ac-3f3f-42db-94d9-65c0822374aa.png)

Read the paper if you want details, there's lots of stuff in this one and the results page doesn't really do it justice.

This paper's architecture could be used to also evaluate the newer VAE and DDPM designs.

---

## [Can we use Common Voice to train a Multi-Speaker TTS system?](https://arxiv.org/pdf/2210.06370.pdf)

The researchers note that Common Voice is full of noise, reverb, incorrect transcripts and thousands of challenging accents/prosodies.

Typically, a TTS model would perform poorly when using the Common Voice dataset.
The reseachers experiment with using a Deep Learning MOS predictor to filter the dataset and find a subset that is suitable for training TTS models.

![image](https://user-images.githubusercontent.com/42448678/217101970-3a9395e8-8c51-41d1-a984-3f184dceb570.png)

---

Results

![image](https://user-images.githubusercontent.com/42448678/217102279-bd1903ec-0b96-4791-943e-a4500fd3168a.png)

They find that training on a Common Voice subset with pred MOS >= 4.0 gives then better quality AND speaker similarity than the TTS model trained on LibriTTS, and much better results than training on unfiltered Common Voice.

I do think the MOS scores are quite low in all results, but this may be due to their 16Khz vocoder or having a slightly different MOS evaluation system.

---

## [PHONEME-LEVEL BERT FOR ENHANCED PROSODY OF TEXT-TO-SPEECH WITH GRAPHEME PREDICTIONS](https://arxiv.org/pdf/2301.08810.pdf)

In this paper the researchers train a ALBERT model on phonemes with MLM task, and also P2G aux task.

![image](https://user-images.githubusercontent.com/42448678/217370051-fb54f9ec-2c63-46be-8a28-498857f91164.png)

They use a similar dataset to previous papers in this area, however for SOME REASON, they don't include MOS values from in domain samples.

![image](https://user-images.githubusercontent.com/42448678/217370425-97ec0cc4-dbc1-421f-bf98-4934340be278.png)

They also ONLY FINE TUNE ONE OF THE MODELS. They fine tune PL-BERT (theirs) but leave MP-BERT completely frozen. Ridiculous paper.

![image](https://user-images.githubusercontent.com/42448678/217370527-0dc13085-524b-4123-97b5-801b9b4b0079.png)

---

![image](https://user-images.githubusercontent.com/42448678/217370817-dbfd942e-0b28-4617-b71b-f241f09833cb.png)

They show that PL-BERT > PL-BERT-without-P2G > BERT > PL-BERT-without-MLM > Nothing. It has been shown in other papers before but it's nice to have another confirmation.

This paper leaves me with more questions than answers (in a bad way). Someone will need to evaluate
- ALBERT against BERT
- fine-tuning against frozen
- P2G + P-MLM vs G+P-MLM

seperately in order to identify if the method outlined in this paper is actually an improvement or not.

---

## [The VoiceMOS Challenge 2022](https://arxiv.org/pdf/2203.11389.pdf)

TODO

---

## [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/pdf/2110.07205.pdf)

TODO

---

## [FastDiff: A Fast Conditional Diffusion Model for High-Quality Speech Synthesis](https://arxiv.org/pdf/2204.09934.pdf)

TODO

---

## [Learning to Maximize Speech Quality Directly Using MOS Prediction for Neural Text-to-Speech](https://arxiv.org/pdf/2011.01174.pdf)

TODO

---

## [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/pdf/2006.11477.pdf)

![image](https://user-images.githubusercontent.com/42448678/217457870-9a4481f4-31af-44c2-b51a-1f173ea79bf3.png)

In this paper the researchers show a new self-supervised technique to pretrain models for ASR task.
They train an encoder to convert raw waveforms into Q (a codebook like VQVAE), and they train a (randomly masked input) Transformer with contrastive loss objective, to output a pred Q that is close to the GT Q and distant from randomly selected Q's from other frames.
To stop the encoder from outputting the same Q for every frame, they use an additional Diversity Loss term.

Conclusion:

> Our model achieves results which achieve a new state of the art on the full Librispeech benchmark for noisy speech. On the clean 100 hour Librispeech setup, wav2vec 2.0 outperforms the previous best result while using 100 times less labeled data. The approach is also effective when large amounts of labeled data are available. 

---

## [SPEAR-TTS: Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision](https://arxiv.org/pdf/2302.03540.pdf)

In this paper ...

They use w2v-BERT to convert audio into semantic (text) tokens.
They use SoundStream to convert audio into and out-of compressed form. Similar to EnCodec VQVAE used by VALL-E.

They use extensive pre-training with each model to significantly improve results.
