# AlphaZero-Inspired Connect Four

A Connect Four agent trained through self-play, using Monte Carlo Tree Search (MCTS) and a PyTorch policy-value neural network.

## Overview

This educational project is inspired by AlphaZero. It combines:

- A complete Connect Four game environment
- A neural network that predicts moves and estimates the value of a board position
- Monte Carlo Tree Search guided by the neural network
- Self-play training
- Adam optimization and L2 regularization

## Files

- `src/game.py`: Connect Four game logic
- `src/network.py`: PyTorch neural network
- `src/mcts.py`: Monte Carlo Tree Search
- `src/train.py`: self-play training
- `models/connect_four_policy_value.pth`: trained model

## Model

The network receives the 6 × 7 board as 42 input values. It has two hidden layers of 128 and 64 units. It outputs:

- probabilities for the 7 columns;
- a value estimate between -1 and 1.

## Installation

Install the required libraries:

`pip install -r requirements.txt`

## Training

Run:

`python src/train.py`

## Note

This is a compact educational project inspired by AlphaZero, not a complete reproduction of the original system.