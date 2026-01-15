## Point Box

Copyright 2025 Steven Mycynek

version: 000140

# A point grouping app

This is a simple machine learning application to show off the basics of TensorFlow.js
and SolidJS. It uses k-means (an unsupervised algorithm) to attempt to place points
into one of two groups based on their centroid. It's nothing fancy, but I wanted
a small application that ran in-browser that could help people get started with ML basics.

This demo is used in a small 'Getting started in ML' talk I'm giving soon.

## Installation

You will need a recent version of [Node.js](https://nodejs.org/en) and either npm or [bun](https://bun.com/),
along with a basic knowledge of bash and modern web deployment.

// Set up and debug

`bun install`

`bun run dev`

// Code styling

`bun run lint`

`bun run format`

// Deployment

`bun run build`

`deploy.sh`

## Other notes

There are a few hidden features to help with debugging.

- Click on the 'Point Box' `<h1>` title to cycle through logging levels
- Click on the 'Agree to disagree' joke to run a short demo script in the console showing
  all the major TensorFlow.js functions we use
- Click on the 'WebGPU/CPU backend' text to show memory usage in the console.

# Live demo

https://stevenvictor.net/pointbox
