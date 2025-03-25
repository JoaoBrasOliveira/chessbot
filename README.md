# Building and efficient chess playing bot

Tags: `Strategy`, `Alpha-beta pruning`, `Ethereum`, `Time-series`.

## Project Overview

**Introduction:** This project, insipired by Kaggle´s FIDE & Google Efficient Chess AI Challenge [https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge], attempted to create a bot that attempts to play like a real chess Grandmaster (GM) with certain resource constraints. 

**Objective:** The goal was to to obtain the most accurate moves possible for chess matches of 10s with 0.1s Simple Delay (where unused time is not banked). However, contrary to AlphaZero and Stockfish engines which follow a brute-force approach (massive pre-computed tables and endless search trees) shifting to a more elegant and efficient design was of the essence.

## Resources
<div align="center">
  <p><strong>Click the image below to access the full report on the chess bot</strong></p>
  <a href="Chessbot_JoaoBrasOliveira.pdf">
    <img src="images/Chessbot.png" alt="Chessbot" width="600" />
  </a>
</div>

## Dataset Description

**Source:** The shared dataset used 'Magnus Carlsen Complete Chess Games 2001-2022' from Kaggle as source. This dataset, is public, and is available at the following link: [https://www.kaggle.com/datasets/zq1200/magnus-carlsen-complete-chess-games-20012022]. You can obtain the transformed version from the gzip-compressed CSV files within the repository.

<!---
[data folder here](./data).
--->

**Structure:** The data is split within three distinct files:
- black.csv.gz - contains The Encyclopaedia of Chess Openings (ECO) codes for the opening played as well as the entire move sequence played within a game where Carlsen played with the black pieces;
- white.csv.gz - contains the same information that black.csv.gz but for games where Carlsen played with the white pieces;
- openings.csv.gz - contains information about all openings and correspondent move sequences.

## To Reproduce

**Codebase:** Complete script for playing a game with the bot is available in this repository.

**Documentation:** The script is documented with very clear comments, guiding the main ideas behind each of the important lines and structures.