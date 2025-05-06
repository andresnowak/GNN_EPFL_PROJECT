This operation (.nonzero()) is generally not differentiable, so it seems dense to sparse causes problems, well it is more about also how you select the values for the edge weights based on the non zero.


## Time series anomaly
- They use Adam trainer with a learning rate of $5e-5$
- They say the best n for averaging is $n=4$ but $n=1$ is also similar so we can say for now doesn't matter
- The window size from smaller also doesn't change to much in their results, so we can say this also doesn't matter (we can use 50)
- They train for 100 epochs