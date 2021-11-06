# Signature Verification Program
This is a handwritten signature verification program trained using a deep convolution neural network with it's visualition implemeted using PyQt5. The user can upload an original image and another image which will be checked with the original to verify whether it is an original signature or a forgery.

## What my application does
Coming soon

## Datasets used
* [CEDAR Signature Dataset](https://paperswithcode.com/dataset/cedar-signature#:~:text=for%20signature%20verification-,CEDAR%20Signature%20is%20a%20database%20of%20off%2Dline%20signatures%20for,thereby%20creating%201%2C320%20genuine%20signatures.)
has signatures in English. It contains 55 individuals’ signature samples. Each individual has 24 genuine signatures and 24 forged signatures.  
**Total: 1320 genuines and 1320 forgeries.**

* [BHSig260 Signature Dataset](https://drive.google.com/file/d/0B29vNACcjvzVc1RfVkg5dUh2b1E/view)
has signatures in Bengali and Hindi.
    * Bengali - It contains 100 individuals’ signature samples. Each individual has 24 genuine signatures and 30 forged signatures.  
    **Total: 2400 genuines and 3000 forgeries.**
    * Hindi - It contains 160 individuals’ signature samples. Each individual has 24 genuine signatures and 30 forged signatures.  
    **Total: 3840 genuines and 4800 forgeries.**

## File structure
- preprocess.py (utility file)
    - class PreProcessing()
    - class SiamesePairs(PreProcessing)
    - class SiameseTriplets(PreProcessing)
    - class SiameseQuadruplets(PreProcessing)
    - class Evaluation(PreProcessing)

- contrastive_utils.py (utility file)
    - euclidean_distance
    - eucl_dist_output_shape
    - contrastive_loss
    - embedding_net
    - build_contrastive_model
    - compute_accuracy_roc
    - evaluation_plots
    - draw_eval_contrastive

- triplet_utils.py (utility file)
    - embedding_net
    - TripletLossLayer
    - build_triplet_model
    - compute_l2_dist
    - compute_probs
    - compute_metrics
    - find_nearest
    - draw_roc
    - draw_eval_triplets

- quadruplet_utils.py (utility file)
    - embedding_net
    - build_metric_network
    - QuadrupletLossLayer
    - build_quadruplet_model
    - compute_l2_dist
    - compute_probs
    - compute_metrics
    - find_nearest
    - draw_roc
    - draw_eval_quadruplets

- CEDAR Verification.ipynb

- visualization.py (GUI)

## How to install
Coming soon

## How to use
Coming soon

## Challenges faced and features to implement in the future
To try a model implementation with a mixed dataset from all 3 scripts.

## License
[MIT](/LICENSE)
