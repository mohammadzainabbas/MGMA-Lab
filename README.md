## Lab 01 - NetworkX 👨🏻‍💻

</br>

### Table of contents

- [Introduction](#introduction)
  * [NetworkX](#network-x)
- [Dataset](#dataset)
  * [SNAP for Python](#snap-for-python)
  * [WormNet v3](#wormnet-v3)
- [Setup](#setup)
  * [Create new enviornment](#create-new-env)
  * [Setup `pre-commit` hooks](#setup-pre-commit)
- [Preprocess](#preprocess)
- [Generate TBOX](#generate-tbox)
- [Generate ABOX](#generate-abox)

---

<a id="introduction" />

#### 1. Introduction

__`Data drives the world.`__ Nowadays, most of the data (_structured_ or _unstructured_) can be analysed as a graph. Today, many practical computing problems concern large graphs. Standard examples include the Web graph and various social networks. The scale of these graphs (_in some cases billions of vertices, trillions of edges_) poses challenges to their efficient processing.

In this lab, we will focus on some basic graph algorithms and see how we can utilise these algorithms to efficiently analyse our data. Since, there exist many similarities between graph theory and network science, you will see us using network science related packages as well. 

<a id="network-x" />

#### 1.1. NetworkX

[NetworkX](https://networkx.org/) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

<a id="dataset" />

### 2. Dataset

For the purpose of this lab, we will use graph datasets. Instead of creating our own graphs (you are more then welcome if you have your own graph datasets), we will use some already existing datasets.

<a id="snap-for-python" />

#### 2.1. SNAP for Python

[Snap.py](https://snap.stanford.edu/snappy/) is a Python interface for SNAP. SNAP is a general purpose, high performance system for analysis and manipulation of large networks. SNAP is written in C++ and optimized for maximum performance and compact graph representation. It easily scales to massive networks with hundreds of millions of nodes, and billions of edges.

SNAP also provides some graph datasets which we will use in this lab. List of available datasets can be found [here](https://snap.stanford.edu/data/index.html).

<a id="wormnet-v3" />

#### 2.2. WormNet v3

[WormNet]() provides a genes data in the form of a network. It's basically a network-assisted hypothesis-generating server for `Caenorhabditis elegans`. List of available datasets can be found [here](https://www.inetbio.org/wormnet/downloadnetwork.php).

WormNet is part of NetBioLab.org 



```bash
conda create -n mgma python=3 -y 
```

```bash
pip install -r requirements.txt
```

```bash
brew install pre-commit
```

or 

```bash
pip install pre-commit
```

Checkout all the `pre-commit hooks` [here](https://pre-commit.com/hooks.html)

Then install the pre-commit hook

```bash
pre-commit install
```


