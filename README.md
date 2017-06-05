# StackInTheFlow



## Table of Contents

- [Introduction](#introuction)
- [Security](#security)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Contribute](#contribute)
- [License](#license)

## Introduction
Programmers increasing rely on an Integrated Development Environment (IDE), which offers capabilities for writing, debugging and testing code. Most developers, ranging from novices to pros, reference outside sources when developing, such as the popular Q&A site StackOverflow. In the 2016 StackOverflow Developer Survey it was observed that 78% of survey participants visited StackOverflow at least once a day, with 58% visiting multiple times a day. The purpose of the StackInTheFlow software development tool is to reduce the amount of time and interruption necessary to gather external information during development. This enables the developer to remain in-the-flow of solving software engineering problems. This tool not only provides standard information retrieval capabilities similar to popular search engines such as Google, but also a feature to auto-generate queries based on the developer’s current work context, extracted from within the IDE. The auto-query feature functions by first extracting features from the IDE, including snippets of source code such as import statements and the current cursor line, as well as, if available, compilation error messages.  From these features candidate query terms are extracted.  These candidate terms are then compared against a dictionary constructed from a dump of all StackOverflow articles.  From this dictionary, various retrieval statistics for pre-retrieval query quality are computed for each term.  The highest ranking terms are then chosen to form a query which is sent to the StackOverflow API and the relevant questions are returned and displayed to the developer. 


## Background



## Install
[StackInTheFlow](https://plugins.jetbrains.com/plugin/9653-stackintheflow)


## Usage


## Contribute

See [the contribute file](contribute.md)!

PRs accepted.

## License

[MIT](../LICENSE)
