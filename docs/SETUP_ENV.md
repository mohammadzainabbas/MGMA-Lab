## Create new enviornment 👨🏻‍💻

### Table of contents

- [Create new enviornment](#create-new-env)
  * [Via conda](#new-env-conda)
  * [Via virtualenv](#new-env-virtualenv)

#

<a id="create-new-env" />

### 1. Create new enviornment

<a id="new-env-conda" />

#### 1.1. Via conda

Before starting further, make sure that you have `conda` (Anaconda) installed (otherwise, create a new env via [virutalenv](#new-env-virtualenv)). We will create a new enviornment for the purpose of our labs:

```bash
conda create -n mgma python=3 -y 
```

and activate it

```bash
conda activate mgma
```

<a id="new-env-virtualenv" />

#### 1.2. Via virtualenv

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

