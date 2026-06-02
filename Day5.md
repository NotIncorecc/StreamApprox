Turns out my image_bind_video teacher was a pure video-visual embedding. ImageBind 
is capable of encoding audio separately, but this code never uses that capability.

- next time we can encode the audio along with the frames to get the real imgbind multimodal embeddings

---

I am only taking one frame as the student model. And In some videos even the audio is not available. Student visual has 512 dimentions whereas default imgbind has 1024 dims and by default imgbind takes only 16 frames as the input. 

- I had an idea, that we can take the middle frame of the video, the frame just after the starting of the video, and just before the ending of the video. Three frames. These would model a type of story for the model to learn

---

- We'll have to re-write all the notebooks again for this optimization, for the first notebook `1_feature_extraction.ipynb` we'll run the notebook completely on kaggle. This time, ensure that the dataset has videos with audio also, not like last time where the audio was missing for 100% of the videos.
Then we'll run the feature extraction overnight on kaggle only, using save and run all cells(see if this is possible)

- After this, the training part of mlp comes in, you have to keep in mind all the issues that came up last time.

- Basically take care of all the other issues after that

---

Start by creating a notebooksV2 folder and start writing the new notebooks. I'll then run the feature_extraction notebook directly on kaggle to avoid data loss due to network or kernel expiry

If there is any issue with my approach, then please correct me, and then lets do this