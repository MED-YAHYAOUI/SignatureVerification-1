# Signature Verification Program
[Check out the Wiki for detailed information](https://github.com/ikathuria/SignatureVerification/wiki)

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

<!--
* [ICDAR Signature Dataset](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011))
has signatures in Dutch and Chinese.
    * Dutch - It contains 10 individuals’ signature samples. Each individual has 24 genuine signatures and 4 forged signatures.  
    **Total: 240 genuines and 140 forgeries.**
    * Chinese - It contains 10 individuals’ signature samples. Each individual has 24 genuine signatures and 12 forged signatures.  
    **Total: 240 genuines and 120 forgeries.**

Why you used the technologies you used
-->

## File structure
- preprocess.py (utility file)
   - class PreProcess()
   - class BinaryClassification(PreProcessing)
   - class SiamesePairs(PreProcessing)
   - class SiameseTriplets(PreProcessing)

- model.py (utility file)
   - def binary_classifier
   - def euclidean_distance
   - def eucl_dist_output_shape
   - def siamese_CNN

- siamese_CNN.ipynb

- binary_classification.ipynb

- visualization.py (GUI)

## How to install
Coming soon

## How to use
Coming soon

## Challenges faced and features to implement in the future
Coming soon

## License
[MIT](/LICENSE)
