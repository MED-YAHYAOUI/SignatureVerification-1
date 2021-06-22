# Signature Verification Program
This is a handwritten signature verification program trained using a deep convolution neural network with it's visualition implemeted using PyQt5. The user can upload an original image and another image which will be checked with the original to verify whether it is an original signature or a forgery.

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

What your application does
Why you used the technologies you used
Some of the challenges you faced and features you hope to implement in the future

## TODO
1. Make Gui
2. Make dataset
3. Make CNN
4. Train CNN
5. Test CNN
-->

### File structure

- create_dataset.ipynb
    - draw_pics
    - label_data
    - build_dataset
    - def get_batch_hard
    - def get_batch_hall
    - def draw_triplets

- siamese_CNN.ipynb
    - Network
        - def build_network
        - class TripletLossLayer
        - def build_model

    - Eval
        - def compute_dist
        - def compute_probs
        - def compute_metrics
        - def compute_interdist
        - def draw_interdist
        - def find_nearest
        - def draw_roc
        - def draw_test_image

- visualization : to test the model (GUI)

## How to install
Coming soon

## How to use
Coming soon

## License
[MIT](/LICENSE)
