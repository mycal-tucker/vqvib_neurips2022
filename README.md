# vqvib_neurips2022
Codebase for VQ-VIB implementation and color experiments based on "Trading off Utility, Informativeness, and Complexity in Emergent Communication" NeurIPS 2022.

Below, we provide a visualization of how VQ-VIB agents communicated in a color reference game.
On the left, we visualize communication vectors using 2D PCA.
On the right, we show the corresponding modemap for how speaker's discretized the color space.
As complexity increases, the number of VQ-VIB tokens increases, and the color categories become more specific.

 ![alt text](gifs/color.gif "Visualization of VQ-VIB communication")

A similar visualization for communication and space discretization in a particle world (see paper for more details) is provided as `gifs/uniform.gif`.

In this repo, we provide the minimal code to allow you to train agents (including VQ-VIB agents) in a color reference game, reproducing the complexity control from the color gif.

# Getting Started

1. Install the requirements listed in requirements.txt.

    ```
    pip install -r requirements.txt
    ```

2. Add the ib-color-naming submodule.

    We depend upon code from Noga Zaslavsky's ib-color-naming github repo. Add it as a submodule via:
    
    ```
    git submodule add https://github.com/nogazs/ib-color-naming ib_color_naming
    ```

3. Lastly, download the IB Color Naming model from Noga Zaslavsky's prior color naming work:

    https://www.dropbox.com/s/70w953orv27kz1o/IB_color_naming_model.zip?dl=1

   Put the contents of the zip file inside a directory called ``models``

    Your directory structure should look like this:
    
    ```
    vqvib_neurips2022/
         -data
           -wcs
         -ib_color_naming
         -models
           -IB_color_naming_model
              -model.pkl
         -saved_data
         -src
           -data
           -models
           -scripts
           -settings
           -utils
    ```

4. Now let's run code. 

    All scripts will be run from the vqvib directory. Not inside ``src`` or anything like that.
    
    If you just want to run a single script, run ``main.py``. Again, we assume you run from vqvib, so the flow on a terminal may look like:
    
    ```
    cd ~/src/vqvib_neurips2022
    python src/scripts/main.py
    ```
    
    That's it! No arguments or anything like that.
    There are essentially arguments hardcoded at the bottom of the script.

Now, to actually see results, look around in the saved_data dir.
There should be a folder created for each epoch in training, with a snapshot of performance and what communication looked like.
E.g., there's PCA of the communication vectors, colored by the color used to generate the communication.
Or there are plots of capacity and accuracy across all epochs up to that point.

There are also two special folders called 'pca' and 'modemaps' that save snapshots during training.
The `pca` folder tracks 2D PCA of learned communication, and associated modemaps.
The `modemaps` folder stores ancillary other modemaps, like the closest human modemap.

You can plot data from multiple trials and architectures using the ``src/scripts/plot_results.py`` script.
Just edit the script to load the relevant models and seeds you'd like to compare.
For example, using VQ-VIB with random seed 1 over 200 epochs (as set up in ``main.py``), we generated a similar range of complexity and informativeness values as WCS languages.
Running ``plot_results.py`` will generate ``info.png`` with the following plot:


 ![alt text](info_example.png "Informativeness and Complexity of VQ-VIB communication, compared to human languages")



## Key settings:
| Name                      | Description                                                                                                                                  |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| comm_dim                  | Dimensionality of communication vectors. For most methods, that allows vectors in, e.g., R^32. For onehot, that specifies how many tokens.   |
| kl_incr                   | By how much to increase the weight of the KL loss every time we update it in training. This is like the annealing step size.                 |                                                                                                                                  |
| obs_noise_var             | Variance of the Gaussian noise used to corrupt the speaker's observation in CIELAB space                                                     |
| plotting_freq             | How frequently (in epochs) to generate plots for the communications. Plotting can slow things down.                                          |
| recons_weight             | Weight for reconstruction loss. Try to decode the speaker's observation from the communication. This is lambda_I in the paper.               |                                                                                                                                        |
| settings.entropy_weight   | This sets by how much to penalize the (approximated) categorical entropy of VQVIB tokens.                                                    |
| Speaker class             | We choose the speaker class by just setting the string at the bottom of main.py. Just comment out the one you want.                          |

## Code Structure

There's one main script, `main.py`, that does the training.
There's another script, `plot_results.py`, that can generate some useful plots across trials, like informativeness and complexity.

The models are all under models. The ``team`` in team.py brings together a speaker, listener, and decoder.
The other modules are mostly just different architectures for different types of communication.


## Integration with IB code
This code integrates with Noga Zaslavsky's IB code (https://github.com/nogazs/ib-color-naming) by adding it as a submodule.
The integration should already be done - we call various methods for calculating gNID, plotting the IB bound, etc.

There appears to be a memory leak issue in calling parts of the IB code related to modemaps; there's a comment to this effect in main.py.
The current "solution" is to just not call that code often.

## Interesting experiments
Below, we list (some of) the ways one can induce interesting new behaviors by changing just a few hyperparameters.

1) Change ``recons_weight`` to measure the effect of our informativeness loss. Greater weights should induce faster convergence to greater complexity and informativeness.
2) Change ``settings.entropy_weight``. In theory, a small weight is sufficient to bias agents to lower-entropy naming patterns. Greater weights induce lower entropy but also start affecting complexity. Further investigation of complexity and categorical entropy losses could be interesting, including by annealing the entropy weight.

## Citation

If you found this code useful, please cite the NeurIPS paper!
```
  @inproceedings{
  tucker2022trading,
  title={Trading off Utility, Informativeness, and Complexity in Emergent Communication},
  author={Mycal Tucker and Roger P. Levy and Julie Shah and Noga Zaslavsky},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=O5arhQvBdH}
}
```
If you have questions, please reach out to Mycal Tucker at mycal@mit.edu.

