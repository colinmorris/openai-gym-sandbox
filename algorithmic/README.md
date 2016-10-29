`policy_gradients.py` is a script that solves a few environments in the [Algorithmic](https://gym.openai.com/envs#algorithmic) category in the OpenAI gym.

Basically semi-homemade policy gradients implementation. Some inspiration drawn from these examples:

- https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
- https://github.com/kvfrans/openai-cartpole/blob/master/cartpole-policygradient.py

Can pretty consistently beat `Copy` and `DuplicatedInput` envs. Could probably beat `RepeatCopy` with some tuning. Unlikely to beat addition environments cause they look hard and this code is currently only set up to handle 1-dimensional input tapes. Could maaaybe beat `Reverse` with some changes, but would be tricky. I've tried it, and it seems really hard to jump out of the local minimum that comes from just blindly guessing the output.

## Hacks

This code is littered with the corpses of failed experiments. The main one that stuck was adjusting the base reward for non-writing actions from 0 to a negative value. Basically, I wanted to discourage dilly-dallying - ping-ponging left-and right a few times then writing an 'A' shouldn't give a higher total reward than just writing the 'A'. I think this is kind of justifiable because the algorithmic environments have (very harsh) time limits. So an action that accomplishes nothing is worse than neutral.

Policy network gets the current character as input and (for some environments) the previously read character and previous action. The previous action input is supplemented with an extra value for 'no previous action', which seems to help.

## Issues

RepeatCopy consistently gets stuck on the policy of 

1. Move to the end of the input tape, copying along the way, stopping when a blank is read
2. Move left and don't write anything
3. Copy what's under the read head (i.e. the last character), and move **right**

I would sure like it to learn to move left at that last step. The problem is that by that point, the policy of moving right when seeing a non-blank character is too hard to dislodge. When my pre-probability layer (y) uses relu, the pre-probability for 'move left' is zero, so there's no gradient. When I remove the relu function and just do wx+b, the left pre-probability is highly negative, and the right one is highly positive, so the chance of actually selecting left is vanishingly small.

It seems like this problem should be solvable with weight penalties, but I haven't had much luck with them. There seems to be a threshold on the l2 scaling such that, anywhere below it, there's no effect, and anywhere above it, learning is just killed entirely.
I think this is because my loss scales with the reward, and the magnitude of rewards varies a lot at different stages of progress (once we've learned the copy policy, the reward associated with the first few actions might be around 2-3, but in the early stages of learning, when the model chances to copy the first character, it'll get a reward of like .65). So for the L2 penalty to be significant enough to affect the later stages, it needs to be high enough that it suppresses learning early on. That's my guess anyways. 

It seems like *that* problem should be solvable by scaling the rewards within an episode/batch to have unit mean/variance, but I've tried that and it seems to hurt for reasons unclear to me. 
