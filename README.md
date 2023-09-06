[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Individual Project Submission

## Can Large Language Models learn physical models?
This repository contains the code for the individual project submission MSc in Artificial Intelligence Course at Imperial College London.
The project was supervised by [Dr Edward Johns](https://www.imperial.ac.uk/people/e.johns). It investigates the architecture of the Transformer Model and attempts to perform supervised learning and physical scenarios in simulation.


[//]: # (# Project Name)

[//]: # ()
[//]: # ([![Build Status]&#40;link-to-build-status-badge&#41;]&#40;link-to-build-status&#41;)

[//]: # ([![License]&#40;link-to-license-badge&#41;]&#40;link-to-license&#41;)

###  Project Description

This projects evaluates the ability of the LLM architecture to learn physical models. It makes use of scene descriptions (in text format) of physical scenarios and attempts to predict the next state of the system. In particular, the project investigates the ability of the LLM to learn the following physical models:
- [x] Free Fall under gravity
- [x] Single Object Collision
- [x] Double Object Collision

With variations in objects' positions, orientations and time over which the observation is made.

These scenarios are simulated using the [arcade library](https://api.arcade.academy/en/latest/) and with a physics engine based in [Pymunk](https://www.pymunk.org/en/latest/). The LLM is trained on the text descriptions of the scenarios and the next state of the system is predicted. The predicted state is then compared to the actual state of the system. The LLM is trained using the [HuggingFace](https://huggingface.co/) library.

The repository amalgamates four repositories which I have worked on for the different parts of the project. While these repositories are not linked, sections of the code have been used in this project. The repositories are:
- **[transformer_from_scratch](https://github.com/Uokoroafor/transformer_from_scratch)**: An implementation of the Transformer architecture from scratch in PyTorch from the paper([Attention Is All You Need](https://arxiv.org/abs/1706.03762)).
- [gpt_from_scratch](https://github.com/Uokoroafor/gpt_from_scratch): An implementation of a smaller version of the GPT model from the paper([Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)) in PyTorch.
- [finetune_llms](https://github.com/Uokoroafor/finetune_llms): A repository that contains code for finetuning and performing In Context Learning on the LLMs.
- [robotics_environment](https://github.com/Uokoroafor/robotics_environment): A repository that contains code for the simulation of the physical scenarios in arcade and pymunk.
## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

Explain how to get your project up and running. This section should include:

### Prerequisites

List any software, libraries, or dependencies that users need to have installed before they can use your project.

### Installation

Provide step-by-step instructions on how to install and set up your project. Include any configuration files that need to be edited and any special setup procedures.

## Usage

Explain how to use your project. Provide code examples, usage scenarios, and any other information that will help users understand how to interact with your code.

## Structure
The repository is structured as follows with the main files in the root directory:
```
├── README.md
├── data
│   ├── collision_data
│   │   ├── collision_data_1.txt
│   │   ├── collision_data_2.txt
│   │   ├── collision_data_3.txt
│   │   ├── collision_data_4.txt
│   │   ├── collision_data_5.txt

```




## Features

List the key features of your project. Bullet points or a brief description of each feature can be helpful.

## Contributing

If you want others to contribute to your project, explain how they can do that. Include guidelines for submitting issues or pull requests, code style conventions, and any other relevant information.

## License

Specify the license for your project. Make it clear how others can use and distribute your code.

## Acknowledgments

If your project uses third-party libraries, APIs, or was inspired by other projects, give credit and provide links to them.

## Frequently Asked Questions (Optional)

If there are common questions or issues that users might encounter, consider including a FAQ section to address them.

## Contact Information

Provide a way for users to contact you or the project maintainers, such as an email address or a link to a discussion forum.

## Support

If you or your team offer support or documentation beyond the README, provide links to those resources.

## Badges (Optional)

You can include badges for build status, code coverage, or other relevant metrics to give users a quick overview of your project's health.

## Examples (Optional)

If your project has a website, demo, or live examples, provide links to them in this section.

## Roadmap (Optional)

If you have plans for the future development of your project, you can include a roadmap or a list of planned features.

## Changelog (Optional)

Keep a record of changes and updates to your project in a separate file and provide a link to it in this section.

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Andrej Karpathy's github repo](https://github.com/karpathy)

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)