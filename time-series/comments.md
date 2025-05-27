This operation (.nonzero()) is generally not differentiable, so it seems dense to sparse causes problems, well it is more about also how you select the values for the edge weights based on the non zero.


## Time series anomaly
- They use Adam trainer with a learning rate of $5e-5$
- They say the best n for averaging is $n=4$ but $n=1$ is also similar so we can say for now doesn't matter
- The window size from smaller also doesn't change to much in their results, so we can say this also doesn't matter (we can use 50)
- They train for 100 epochs
- Maybe we can use Graph transformers for the graph learner part
- The attention module in the paper is to learn the signal representations (like to force the model to know what they are, so it learns to reconstruct it), and it could be be possible to do it with the coarse grained labels, but the idea is very different, so probably in reality it isn't possible to do the idea with coarse graiend lables


The name of resnet with LSTM model is not SMOTE, that was the method they used for oversampling to reduce overfitting for the majority class, but we will leave the name on the files as it is easily identifiable.

**It seems it is necessary to use a better dataset division as BILSTM during training reported a validation accuracy of 0.85 but when doing the submission it was only 0.69**