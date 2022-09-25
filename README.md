## Lab Work @ Massive Graph Management and Analytics 👨🏻‍💻

### Table of contents

- [Introduction](#introduction)
- [About the repo](#about-repo)
- [Labs](#labs)
- [Setup](#setup)
  * [Create new enviornment](#create-new-env)
    * [Via conda](#new-env-conda)
    * [Via virtualenv](#new-env-virtualenv)
  * [Setup `pre-commit` hooks](#setup-pre-commit)
<!-- - [Preprocess](#preprocess)
- [Generate TBOX](#generate-tbox)
- [Generate ABOX](#generate-abox) -->

#

<a id="introduction" />

### 1. Introduction

__`Data drives the world.`__ Nowadays, most of the data (_structured_ or _unstructured_) can be analysed as a graph. Today, many practical computing problems concern large graphs. Standard examples include the Web graph and various social networks. The scale of these graphs (_in some cases billions of vertices, trillions of edges_) poses challenges to their efficient processing.

In this course, we will focus on some basic graph algorithms and see how we can utilise these algorithms to efficiently analyse our data. Since, there exist many similarities between graph theory and network science, you will see us using network science related packages as well. 

<a id="network-x" />

#### 1.1. NetworkX

[NetworkX](https://networkx.org/) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

#

<a id="dataset" />

### 2. Dataset

For the purpose of this lab, we will use graph datasets. Instead of creating our own graphs (you are more then welcome if you have your own graph datasets), we will use some already existing datasets.

<a id="snap-for-python" />

#### 2.1. SNAP for Python

[Snap.py](https://snap.stanford.edu/snappy/) is a Python interface for SNAP. SNAP is a general purpose, high performance system for analysis and manipulation of large networks. SNAP is written in C++ and optimized for maximum performance and compact graph representation. It easily scales to massive networks with hundreds of millions of nodes, and billions of edges.

SNAP also provides some graph datasets which we will use in this lab. List of available datasets can be found [here](https://snap.stanford.edu/data/index.html).

<a id="wormnet-v3" />

#### 2.2. WormNet v3

[WormNet](https://www.inetbio.org/wormnet/) provides a genes data in the form of a network. It's basically a network-assisted hypothesis-generating server for `Caenorhabditis elegans`. List of available datasets can be found [here](https://www.inetbio.org/wormnet/downloadnetwork.php).

WormNet is part of [YONSEI Network Biology Lab](https://netbiolab.org/w/Welcome). You can check out other available software/data [here](https://netbiolab.org/w/Software).

#

<a id="setup" />

### 3. Setup

If you want to follow along, make sure to clone and checkout to lab's branch:

```bash
git clone https://github.com/mohammadzainabbas/MGMA-Lab.git
cd MGMA-Lab/
git checkout lab1
```

<a id="create-new-env" />

#### 3.1. Create new enviornment

<a id="new-env-conda" />

##### 3.1.1. Via conda

Before starting further, make sure that you have `conda` (Anaconda) installed (otherwise, create a new env via [virutalenv](#new-env-virtualenv)). We will create a new enviornment for the purpose of our labs:

```bash
conda create -n mgma python=3 -y 
```

and activate it

```bash
conda activate mgma
```

<a id="new-env-virtualenv" />

##### 3.1.2. Via virtualenv

You can create your virtual enviornment without conda as well. In order to do that, make sure that you have [`virtualenv`](https://pypi.org/project/virtualenv/) installed or else, you can install it via:


```bash
pip install virtualenv
```

Now, create your new enviornment called `mgma`

```bash
virtualenv -p python3 mgma
```

and then activate it via

```bash
source mgma/bin/activate
```

#

Once, you have activated your new enviornment, we will install all the dependencies of this project:

```bash
pip install -r requirements.txt
```

<a id="setup-pre-commit" />

#### 3.2. Setup `pre-commit` hooks

Git hook scripts are useful for identifying simple issues before submission to code review. We run our hooks on every commit to automatically point out issues in code such as _missing semicolons_, _trailing whitespace_, and _debug statements_. Checkout `pre-commit` hook [site](https://pre-commit.com/index.html) for more details.

You can setup `pre-commit` hook by running:

```bash
brew install pre-commit
```

or if you prefer `pip`

```bash
pip install pre-commit
```

You can have a look at some basic pre-commit hooks that I have added under [pre-commit-config.yml](https://github.com/mohammadzainabbas/MGMA-Lab/blob/lab1/pre-commit-config.yml) file. You can checkout all the `pre-commit hooks` [here](https://pre-commit.com/hooks.html). And add any hook that you like in this file.

Now, install the pre-commit hook by running:

```bash
pre-commit install
```

you will see

```txt
pre-commit installed at .git/hooks/pre-commit
```

This means pre-commit hook were installed successfully.

#

