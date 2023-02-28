# Contributing to RouteNet-Fermi

Thank you for your interest in contributing to RouteNet-Fermi! If you wish to contribute, please abide by our branching 
system.

## How to contribute
The preferred workflow to send contributions to the project is:
* Fork the project on Github. This will create a repository under your Github username with a copy of the code, 
with write access for you.
* Clone the forked repository to your machine and work on the feature/bugfix/etc.
* Commit the changes to your repo, making sure you previously set up git like this.
* Create a patch against the corresponding banch, and send it for comments on the developer mailing list (see the 
description of our **branching system** for more information).
* Modify your code, according to feedback received (if the case).
* Send a pull request from your github repo. If you don't want to use Github, that's OK too, we accept patches directly
sent to the developer mailing list as well. In this case, we will credit you in the commit message.

## Managing branches in RouteNet-Fermi

There are three main branches in Ignnition:

- **main**: The main branch of RouteNet-Fermi. This branch contains the latest stable release.
- **nightly**: The nightly branch of RouteNet-Fermi. This branch contains the small incremental updates that 
are released in a constant release cycle.
- **development**: The development branch of RouteNet-Fermi. This branch contains future features that may drastically 
change how RouteNet-Fermi is used. As such, this branch is expected to be unstable and should not be used for any other
use than alpha testing.
  
There are three different kinds of development branches:

- **hf-(name)**: This branch contains hotfixes (i.e. fixes deemed urgent enough to be included directly into the main 
branch without having to wait to the end of the development cycle). These branches should start from the **main** 
branch and its Pull Request (PR) should be directed at the **main** branch. The hotfix version should be increased whenever one of these
branches is created.
  
- **bf-(name)**: This branch contains bugfixes (i.e. fixes not deemed urgent enough to be included directly into the 
main branch and that can wait until the development cycle ends). These branches should start from the **ignnition-nightly**
branch and its PR should be directed at the **ignnition-nightly** branch.

- **ft-(name)**: This branch contains new features. These branches should start from the **ignnition-nightly** branch
and its PR should be directed at the **ignnition-nightly** branch.
  
- **dev-(name)**: This branch contains new features. These features are expected to contain deeper changes than those in
the **ft-(name)** branches. These branches should start from the **development** branch and its PR should be directed to
the **development** branch.

## Versioning

The *_version.py* file inside the *RouteNet-Fermi* folder contains the version of the package. Versions follows the 
format '**x.y.z**', where '*x*' is the **major** version, '*y*' is the **minor** version, and '*z*' is the **fix** 
version.

The **fix** version is increased with every **hf-(name)** branch. The expected workflow should be the following: create 
a new hotfix branch, increase the fix version and create a PR to the **main** branch. After that, the performed changes 
should be ported to the nightly branch with a cherry-pick to ensure that the hotfix is also present there.

The **minor** version is increased at the end of every development cycle. The expected workflow should be the following:
at the beginning of every development cycle, increase the minor version of the **ignition-nightly** branch. At the end 
of the development cycle, create a PR to the **main** branch to bring all the changes made during the cycle to the
next release. 
Increasing this version resets the **fix** version.

The **major** version is increased with every new **development** branch. The expected workflow should be the following:
whenever a new **development** branch is created, increase the major version. Whenever it is deemed necessary, create a 
PR to the **main** branch to bring all the changes to the next release.
Increasing this version resets the **minor** and **fix** versions.

# Software maintainers
The current maintainers of the code are listed below:
* Miquel Ferriol-Galmés (miquel.ferriol@upc.edu)