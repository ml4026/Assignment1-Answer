# Assignment1-Answer

## Introduction

### Background
Winter is here. You and your friends were tossing around a frisbee at the park when you made a wild throw that left the frisbee out in the middle of the lake. The water is mostly frozen, but there are a few holes where the ice has melted. If you step into one of those holes, you'll fall into the freezing water. At this time, there's an international frisbee shortage, so it's absolutely imperative that you navigate across the lake and retrieve the disc. However, the ice is slippery, so you won't always move in the direction you intend. The episode ends when you reach the goal or fall in a hole.

### Mathematical Model
The states form a 4 * 4 grid. There are 4 kinds of states. "S" is the safe starting point, "F" represents frozen surface, which is safe as well. "H" represents a hole. You "fall to your doom" if you enter "H" states. "G" is your goal where the frisbee is located.<br />
SFFF<br />
FHFH<br />
FFFH<br />
HFFG<br />
The indices of states are shown below.<br /> 
00 01 02 03<br />
04 05 06 07<br />
08 09 10 11<br />
12 13 14 15<br />
At each step, you can take 4 actions: "LEFT", "DOWN", "RIGHT", "UP" represented by indices 0 to 3 respectively. Your next state is then given by the environment. The episode ends when you reach the goal or fall in a hole. You receive a reward of 1 if you reach the goal, and 0 otherwise.