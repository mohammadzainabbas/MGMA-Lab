## Lab 01 - NetworkX 👨🏻‍💻

</br>

### Table of contents

- [Introduction](#introduction)
  * [NetworkX](#network-x)
  * [Apache Jena](#apache-jena)
  * [Ontology](#ontology)
    * [TBOX](#tbox)
    * [ABOX](#abox)
- [Setup](#setup)
  * [Create new enviornment](#create-new-env)
  * [Setup `pre-commit` hooks](#setup-pre-commit)
- [Dataset](#dataset)
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

To install it, first make sure that you have setup your enviorment well.



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


