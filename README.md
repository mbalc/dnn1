# DNN task 1

Solution for assignment from Deep Neural Networks
I got pretty bad results so this should be possible to optimize

Dataset was based on fruits-360,
task involved implementinng basic conv-net with own implementation of batch-norm

## outputs
`pixelwise` is heatmap of input gradients per pixel
`occluded` is "importance of a picture part", ie. where the net focuses to distinguish. 
This one is obtained by greying out squares surrounding a pixel, then calculating loss of such picture
