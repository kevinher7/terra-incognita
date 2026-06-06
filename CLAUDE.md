# Terra Incognita

The current repo is a training system for the model that is part of a failure mode observability system.

This multi repo project has the following objectives
- Experience creating multiple systems with AI agent tools to push its limits
- Learn about the best practices when training a model with MLOps Tools
- Learn about the challenges of training a model, as well as the unique challenges of serving it
- Make the whole system end to end as observable as possible, this means great observability via wide event logging

## Building this project

As this is a learning excercise to understand the challenges of creating complex computer vision systems with great observability, please make sure that we reach a common understanding on each of the things we build, be explicit about what decision you are taking and the WHY you do it that way.

Always chose the simpler option rather than an overengineered one. If the time does arise when a slightly overengineered option is a genuine possibility, then open a discussion with me so that we can get to an understanding ot this.

The current repo is for training the system, but rather than achieving the highest possible accuracy, it's more about learning about best practices around the training process using MLOps tools, as well as serving those models and of course, complete a end to end CV failure mode system.


## Observability

One of the main goals is to have great visibility. Therefore, always make a point of the discussion "what kind of data will be useful (even if remotely) for debugging later on". There is no such a thing as too much information, let's aim to be able to aggregate on our wide events to be able to debug issues with ease.

## SSoT

The single source of truth for cross-repo coordination, shared contracts, and
each repo's design lives in `.plans/`, a **read-only mirror** of the
[`terra-carta`](https://github.com/kevinher7/terra-carta) repo (vendored via
`git subtree`). When a question touches another repo's design or a shared
interface, read it from `.plans/` rather than guessing or restating it here.
