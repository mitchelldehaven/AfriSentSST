# AfriSentSST
This repo contains the code for my approach to the SemEval 2023 AfriSenti task. For SubTask B and SubTask C, my approach ranked 5th and 7th respectively among the ~ 35 participants participating in the task. I initially was going to write a short paper (4 pages) for the task, but found that I didn't really do enough to stretch the system description to even 4 pages, so I'll just pubish the general ideas and code here if others are interested. The name I gave the system was `AfriSentSST` because that's essentially all I did. I didn't use any sentiment lexicons or additional resources to improve performance, simply SST via translated pseudo-labelled English Tweets. 

Much of the current repo is from the cloned repo of the task:
https://github.com/afrisenti-semeval/afrisent-semeval-2023

My source code for building the models is under `src/`

Due to the restrictions of redistributing Twitter's data, I won't be redistributing the Tweets used for generating the SST data for finetuning.

# General Idea
The 3 subtasks for AfriSenti were:
- SubTaskA: 12 languages, 12 monolingual models
- SubTaskB: 12 languages, 1 multilingual models
- SubTaskC: 2 languages, 2 models, zero-shot learning (no training data provided for languages)

I used 2 models for each task. The first being XLM-R, which is probably the most common multilingual pretrained model. The second was NLLB-200, using only the encoder stack. The idea behind using NLLB-200 was simply an assumption that the task of neural machine translation provides better alignment along tokens across languages as opposed to simple multilingual masked pretraining. Based on this assumption, I assumed that NLLB-200 would do better for SubTaskB and SubTaskC for sure, since there would be more sharing across languages during finetuning due to better alignment. 

Additionally, I wanted to utilize additional unsupervised Tweets somehow. The initial idea was to use continued pretraining, so I gathered about ~580,000,000 Tweets from the free Twitter 1% stream from 2018-2019. However, when trying to get additional data for Amharic, I found that there was only about 10,000-20,000 Amharic Tweets in this entire set. Amharic was one of the higher resource languages featured in AfriSenti and also has a very unique script which makes finding the Tweets a lot easier. I would likely need an order of magnitude more unsupervised data than I had finetuning data provided for the task to make a difference and it simply wasn't there for Amharic and finding data for the other languages that had latinized scripts would have been much harder. 

Instead, the approach taken was to take about ~ 5,000,000 English tweets and pseudo-label them. Using the previous sentiment analysis tasks from SemEval 2013-2017 I finetuned an English model. Using this English model, I pseudo-labelled the English Tweets, using a 0.7 threshold (picked somwhat arbitrarily) and other filters to get ~ 2,700,000 pseudo-labelled Tweets. NLLB-200 was trained on either the exact or a close language for 13 of the 14 total languages, the only exception being Nigrian Pidgin. Using NLLB-200, I translate the pseudo-labelled English Tweets into each of the target languages. I do some filtering here, like removing handles and URLs, but that's about it. I then use this data for finetuning the models. I finetune first of the SST data, then finetune on the data provided. I do this sequentially rather than jointly as initial experiments seemed to indicate that there was some kind of mismatch between the data I generated via pseudo-labelling and the data provided for the task. What caused this mismatch, I'm not sure. It could be noisy translations from NLLB-200, it could have been the sampling procedure used for generating the AfriSenti datasets, etc. I didn't really explore what the could have been the cause here.

For the monolingual approach, it appeared that XLM-R performed much better. This was a bit surprising for some languages, since NLLB-200 has individual tokenizers per languages, so I assumed it would do a better job on some of those, however XLM-R did better on the monolingual task for 10 of the 12 langauges.

For the multilingual approach NLLB-200 did much better than XML-R, which aligned with my expectations. The NLLB-200 model got interupted during training and due to timing, I didn't have time to restart training, so it only ran 1/2 of an epoch on the SST data.

For zero-shot, I only tried using NLLB-200. I simply used the model for SubTaskC and ran the data through. It wasn't fully zero shot, as both these languages were representing in the SST data, but here was no true training data used for this part. 

# Results
For the SubTasks, my ranks were as follows (I didn't compute SubTaskA's and they didn't release a ranking or a parsable format for me to compute it myself, so I just guess it by my relative rank for each individual language):
- SubTaskA: ~10th out of 35 participants
- SubTaskB: 5th out of 33 participants
- SubTaskC: 7th out of 29 participants

# Todo
- Full table results pending... when I get around to organizing my results.
- Code cleanup. I figured I should get the code up then worry about cleaning it later.
- Define requirements.txt. For the most part, it is just Pytorch, Pytorch-Lightning, Transformers, and Scikit-Learn, although I still have the conda environment so I should be able to just dump the requirements.

# Potential future improvements
- I only submitted a single model to be ran against the test set for each task. Given the imbalance in the training set across langauges, one could have used class-weighting and tested on the test set per language if this improved performance. I didn't both testing multiple models on the test sets, so I'm not sure if this would have helped. I think I have 
- Use more SST data: I still have ~ 580,000,000 Tweets, which would be primarily English. One could take this approach and just scale up the data. Since I only used about 5,000,000 Tweets, this would be a 100x increase in terms of what I used. I don't currently have the hardware to support this (just doing what I did from the 5,000,000 Tweets was estimated to take ~ 56 hours iirc for finetuning on my 3090). 
