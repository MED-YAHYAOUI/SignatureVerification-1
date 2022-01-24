<p align="center">
    <a href="resources\logo.png" rel="noopener">
        <img width=200px height=200px src="resources\logo.png" alt="Project logo">
    </a>
</p>

<h3 align="center">Offline Signature Verification</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/ikathuria/SignatureVerification.svg)](https://github.com/ikathuria/SignatureVerification/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/ikathuria/SignatureVerification.svg)](https://github.com/ikathuria/SignatureVerification/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center">
    This is am offline signature verification program trained using a deep siamese convolution neural network. The user can upload an original image and another image which will be checked with the original to verify whether it is an original signature or a forgery.
    <br>
    <a href="https://github.com/ikathuria/SignatureVerification/wiki">Read More - Wiki</a>.
    <br> 
</p>

## 📝 Table of Contents
- [About](https://github.com/ikathuria/SignatureVerification/wiki)
- [Getting Started](#getting_started)
- [Built Using](#built_using)
- [TODO](TODO.md)

## 🏁 Getting Started <a name = "getting_started"></a>
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. First, clone the repository.

### Prerequisites
The next step is to download the required packages which can be done with, requirements.txt which contains all the python packages needed to run the program.  
Make sure you're in the repository directory and then run the following command:

```
pip install -r requirements.txt
```

## ⛏️ Built Using <a name = "built_using"></a>
- [Python](https://www.python.org/)
- [Kaggle](https://www.kaggle.com/)
- [CEDAR Signature Dataset](https://paperswithcode.com/dataset/cedar-signature#:~:text=for%20signature%20verification-,CEDAR%20Signature%20is%20a%20database%20of%20off%2Dline%20signatures%20for,thereby%20creating%201%2C320%20genuine%20signatures.)
has signatures in English. It contains 55 individuals’ signature samples. Each individual has 24 genuine signatures and 24 forged signatures.  
Total: 1320 genuines and 1320 forgeries.
- [BHSig260 Signature Dataset](https://drive.google.com/file/d/0B29vNACcjvzVc1RfVkg5dUh2b1E/view)
has signatures in Bengali and Hindi.
    - Bengali - It contains 100 individuals’ signature samples. Each individual has 24 genuine signatures and 30 forged signatures.  
    Total: 2400 genuines and 3000 forgeries.
    - Hindi - It contains 160 individuals’ signature samples. Each individual has 24 genuine signatures and 30 forged signatures.  
    Total: 3840 genuines and 4800 forgeries.
