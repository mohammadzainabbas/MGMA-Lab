## Lab Work @ Massive Graph Management and Analytics 👨🏻‍💻

### Table of contents

- [Introduction](#introduction)
- [About the course](#about-course)
  * [Main Topics](#main-topics)
- [Labs](#labs)
  * [Lab 01 - NetworkX](#lab-1)
  * [Lab 02 - Influence Models](#lab-2)
  * [Lab 03 - Introduction to Spark](#lab-3)
  * [Lab 04 - PageRank in Spark](#lab-4)
- [Setup](#setup)
  * [Create new enviornment](#create-new-env)
  * [Setup `pre-commit` hooks](#setup-pre-commit)


#

<a id="introduction" />

### 1. Introduction

__`Data drives the world.`__ Nowadays, most of the data (_structured_ or _unstructured_) can be analysed as a graph. Today, many practical computing problems concern large graphs. Standard examples include the Web graph and various social networks. The scale of these graphs (_in some cases billions of vertices, trillions of edges_) poses challenges to their efficient processing.

#

<a id="about-course" />

### 2. About the course

Data we produce or consume has increasingly networked structures which have grown in complexity in different domains such as `biology`, `social networks`, `economy`, `communication` and `transport networks`. The need to process and to analyze such data carries out the emergence of network science research community to define algorithms which allow to characterize such complex structures, to understand their topology, their evolution and to interpret the underlying phenomena. 

Besides, the distributed storage and parallel computation technologies offer specific tools for networks based on large-scale graph processing paradigms such as `MapReduce` and `Pregel` of Google.

The purpose of this course is to study the main algorithms and their implementation on artificial and real data in a distributed environment.

<a id="main-topics" />

#### 2.1. Main Topics

- [x] Preliminaries, Typology of graphs, Graph analytics measures
- [x] Basic algorithms: Random walk and Page Rank
- [x] Label propagation, Community detection, Influence maximisation
- [x] Graph analytics & Deep Learning

#

<a id="labs" />

### 3. Labs

The main aim of this repository is to keep track of the work we have done in __Massive Graph Management and Analytics (MGMA)__ labs. During this course, we will focus on some basic graph algorithms and see how we can utilise these algorithms to efficiently analyse our data. Since, there exist many similarities between graph theory and network science, you will see us using network science related packages as well.

#

<a id="lab-1" />

#### 3.1. Lab 01 - NetworkX

[NetworkX](https://networkx.org/) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

Please checkout lab's details [here](https://github.com/mohammadzainabbas/MGMA-Lab/tree/main/src/lab1) 

<a id="lab-2" />

#### 3.2. Lab 02 - Influence Models

<!-- [NetworkX](https://networkx.org/) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. -->

Please checkout lab's details [here](https://github.com/mohammadzainabbas/MGMA-Lab/tree/main/src/lab2) 
#

<a id="lab-3" />

#### 3.3. Lab 03 - Introduction to Spark

In this lab, we will be introduced to the powerful big data processing framework, [`Spark`](https://spark.apache.org/). [`Apache Spark`](https://spark.apache.org/) is an `open-source`, `distributed computing system` that can `handle large amounts of data` with ease. It is built on top of the [`Hadoop ecosystem`](https://www.edureka.co/blog/hadoop-ecosystem) and can process data in a variety of formats, including `HDFS`, `HBase`, and `local file systems`.

One of the most popular ways to use Spark is through [`PySpark`](https://spark.apache.org/docs/latest/api/python/), which is the Python library for Spark. [`PySpark`](https://spark.apache.org/docs/latest/api/python/) allows us to use the powerful data processing capabilities of Spark with the ease and familiarity of Python. In this lab, we will be using `PySpark` to work with data and perform various operations on it.

We will start by setting up a Spark environment and then move on to loading data into Spark. We will then learn how to perform basic operations such as filtering, mapping, and reducing data. We will also learn how to use the DataFrame API to perform more advanced operations on our data.

By the end of this lab, you will have a solid understanding of how to use `PySpark` to process and analyze large amounts of data. So, let's get started!

Please checkout lab's details [here](https://github.com/mohammadzainabbas/MGMA-Lab/tree/main/src/lab3) 

#

<a id="lab-4" />

#### 3.4. Lab 04 - PageRank in Spark

In this lab, we will be implementing the [`PageRank`](https://en.wikipedia.org/wiki/PageRank) algorithm in [`PySpark`](https://spark.apache.org/docs/latest/api/python/). PageRank is an algorithm used to measure the importance of a webpage within a set of webpages. It was originally developed by Google and is used as a key component in the Google search engine.

The basic idea behind the PageRank algorithm is that a webpage is considered important if it is linked to by other important webpages. The algorithm assigns a score to each webpage, which is based on the number and importance of the webpages that link to it.

We will be using PySpark to implement the PageRank algorithm in this lab. We will start by loading the data and creating a graph representation of the webpages. We will then implement the PageRank algorithm using the Spark RDD (Resilient Distributed Dataset) API and the DataFrame API.

By the end of this lab, you will have a solid understanding of how to use Spark to implement the PageRank algorithm and how to use the RDD and DataFrame APIs to perform data processing tasks. So, let's get started!

Please checkout lab's details [here](https://github.com/mohammadzainabbas/MGMA-Lab/tree/main/src/lab4) 

#

<a id="setup" />

### 4. Setup

If you want to follow along with the lab exercises, make sure to clone and `cd` to the relevant lab's directory:

```bash
git clone https://github.com/mohammadzainabbas/MGMA-Lab.git
cd MGMA-Lab/src/<lab-of-your-choice>
```

> For e.g: if you want to practice lab # 1, then you should do `cd MGMA-Lab/src/lab1`.

<a id="create-new-env" />

#### 4.1. Create new enviornment

Before starting, you may have to create new enviornment for the lab. Kindly, checkout the [documentation](https://github.com/mohammadzainabbas/MGMA-Lab/blob/main/docs/SETUP_ENV.md) for creating an new environment.

#

Once, you have activated your new enviornment, we may have to install all the dependencies for a given lab (kindly check if `requirements.txt` file exists for a given lab before running the below command):

```bash
pip install -r requirements.txt
```

<a id="setup-pre-commit" />

#### 4.2. Setup `pre-commit` hooks

In order to setup `pre-commit` hooks, please refer to the [documentation](https://github.com/mohammadzainabbas/MGMA-Lab/blob/main/docs/SETUP_PRE-COMMIT_HOOKS.md).

#

