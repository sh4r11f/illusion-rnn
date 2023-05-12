# TODO

- There's only a limited number of images in the entire dataset: X number of stimulus sets, 3 trial (motion) type per 
    stimulus set, T number of timepoints where the image changes (lots of redundancy here too in terms of fixation
    period etc.). It'd be faster to run the CNN on image "types" first and then compile them back together as input
    to the RNN. 

- Make the fixation period look like the rest of the experiment (i.e. the same as the stimulus period) by repeating
    the last image of the stimulus period.
- Change parameter file to YAML format.
- Change environment size to work with any resolution.