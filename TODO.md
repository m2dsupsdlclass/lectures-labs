# Improvements for next time we run this class


## Technical Improvements

- Update code to work with TF 1.0+ and Keras 2.0+
- Reduce the size of the repo by compressing png better and reducing
  the resolution of very large images.
- (Semi)-automate the rendering of the notebooks somehow...

## General improvements

- Systematically include architecture diagrams in notebooks,
  possibly using using the name of Keras classes in nodes.


## Lab #2 (embeddings and recsys)

- Highlight the importance of time-based cross-validation splits and
  other cross-validation splits: to measure the ability of the model to
  generalize either to the future, to new users or to new items.

- Lecture: do not tie explicit feedback to regression metrics, it would
  be possible to use ranking metrics for explicit feedback.

- Embedding diagrams should include the one-hot vector of the data
  points that is multiplied with the embedding matrices to emphasize the
  fact that embedding matrices holds model parameters and not training
  data (the training sample is the one-hot vector representation of the
  user / item).


## Lab #3 (convolutions)

- Introduce convolution using the Keras API only. This can be done
  by using using the following test construct:
  `Sequential([Convolution2D(**conv_params)]).predict(test_img)`

- Add exercises that ask to estimate the parameter size and shape of
  the output of a stack of Conv + Pooling layers and then ask to use
  keras to write a program to empirically check the results.


## Lab #4 (advanced convnets)

- Factorize out the PASCAL VOC annotation extraction in a helper module
  to hide the complexity of setting up the learning task so as to focus
  the students attention on the model architecture and less on the
  complexity of annotation preprocessing.

- Introduce the matplotlib utility to display bounding box earlier so as
  to display samples from the training set before introducing the ground
  truth representation of the labels, the IoU think and the models
  themselves.

- Use the matplotlib utility to display pairs of overlapping boxes and
  their IoU value on a random image. On the of the box can be the true
  annotation from the sample and the other an arbitrary overlapping
  bbox.

- Do a cleaner train / validation split, once and for all at the
  beginning of the notebook.


