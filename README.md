# RHINE: a curriculum-trained PPO agent for Bomberman

Author: Qin Yan

<p align="center"><img src="media/rhine_demo.gif" width="480" alt="RHINE playing against three rule_based_agents"></p>
<p align="center"><em>One game against three <code>rule_based_agent</code>s. Individual games vary widely. Average results are given under <a href="#results">Results</a>.</em></p>

RHINE is a PPO agent for four-player Bomberman that acts on 31 engineered features under a rule-based safety mask. Trained from scratch through a staged curriculum, it beats three scripted `rule_based_agent`s, scoring 6.33 points per game against 5.04 for the best opponent and finishing alone in first place in 53% of games.

It was developed as the author's individual part of a group project in the course *Machine Learning Essentials* at Heidelberg University (summer semester 2026). This repository contains only the author's own work.

## Method

- **PPO actor-critic.** A multilayer perceptron with a shared trunk of two 64-unit tanh layers, a policy head over the six actions and a value head, trained with the clipped PPO objective and generalized advantage estimation.
- **31 engineered features.** Computed from breadth-first-search distances, blast coverage and time-dependent danger. No feature encodes a recommended action.
- **Rule-based action mask.** Removes illegal actions and actions that lead to a known, unavoidable death. The policy chooses among the remaining ones, in training and in play.
- **Oscillation breaker.** An evaluation-only refinement of the mask that stops the agent from stepping back and forth between two tiles. It is never active during training.
- **Four-task curriculum.** The course defines four tasks: coins on a board without crates, crates and bombs, passive and coin-collecting opponents, and stronger opponents such as the `rule_based_agent`. The training stages within these tasks were designed by the author, for example four opponent stages in Task 3 with one and then three opponents of each kind. Every task and stage was trained from a random initialization, without weight transfer.

## Results

Task 4 against three `rule_based_agent`s on the `classic` scenario: 100 games per training seed (evaluation seed 1000), mean ± sample standard deviation over three training seeds, oscillation breaker on.

| Agent | Score | Kills | Self-kill rate | Killed-by-opponent rate | Oscillation fraction |
|---|---|---|---|---|---|
| Task 4 agent | 6.33 ± 0.76 | 0.58 ± 0.19 | 0.113 ± 0.025 | 0.130 ± 0.026 | 0.007 ± 0.006 |
| Stage D agent (baseline) | 5.00 ± 0.54 | 0.34 ± 0.07 | 0.153 ± 0.104 | 0.107 ± 0.006 | 0.007 ± 0.012 |

In the same games, the best of the three opponents averaged 5.04 points per game, and RHINE finished alone in first place in 53% of games (63% including ties). Under a second evaluation seed (2000), RHINE scored 6.11 against 4.84 and finished alone in first place in 52% of games.

The baseline is the final Stage D agent (the last stage of Task 3), evaluated zero-shot under the same settings. Besides its training opponents (three `coin_collector_agent`s instead of three `rule_based_agent`s), it lacks the three opponent-proximity features added in Task 4 and was trained with a mask rule, dropped in Task 4, that removed BOMB once the board was cleared. The comparison therefore measures the combined effect of these changes.

All evaluations in the report were run with the framework's per-step thinking-time limit disabled, so that load fluctuations on the evaluation machine could not force WAIT actions. The decision time of the final agent was measured separately in 20 games: the mean `act()` time was 15.8 ms and the maximum 125.2 ms, well below the 0.5 s limit.

## Key findings

- Oscillation turned out to be a limitation of the Markovian state rather than of the reward. The evaluation-time breaker removed it at no cost in score, although it is a rule, not a learned skill.
- Reward shaping destabilized learning on the dense coin board but was indispensable once crates made rewards sparse.
- The Task 4 agent scored significantly higher than the Stage D agent, with more kills per game (0.58 instead of 0.34) and no significant change in the self-kill and killed-by-opponent rates.
- Most remaining deaths come from losing a same-step contest for a tile, which the agent cannot control.
- The score differs between training seeds by more than it differs between the Task 4 agent and the Stage D agent.

## Repository layout

```
agent_code/
  rhine/            Agent code, the final checkpoint (models/), training and evaluation scripts (scripts/) and tests (tests/)
media/              Demo animation used in this README
patches/
  agents_logging.patch   Logging fix applied to the framework's agents.py by setup.sh
report/
  rhine_report.pdf  Full report on RHINE
  rhine_report.tex  LaTeX source of the report
  figures/          Figures used in the report
setup.sh            Downloads the framework into framework/ and links the agent into it
requirements.txt    Python dependencies (install PyTorch separately, see below)
LICENSE             MIT license for the code in this repository
```

## Installation

Requires Python 3.12 and git. Tested on Linux (WSL2).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
bash setup.sh
```

`setup.sh` clones the course framework into `framework/`, checks out a fixed commit, links `agent_code/rhine` into `framework/agent_code/` and applies `patches/agents_logging.patch`. It can be run again safely.

## Usage

All commands are run from `framework/`:

```bash
cd framework
```

One game of RHINE against three `rule_based_agent`s, with and without the GUI:

```bash
python main.py play --agents rhine rule_based_agent rule_based_agent rule_based_agent
python main.py play --agents rhine rule_based_agent rule_based_agent rule_based_agent --no-gui --n-rounds 1
```

In GUI mode the framework waits for a key press (an arrow key, Enter, Space, Q or Esc) after each round before starting the next one. RHINE loads `agent_code/rhine/models/task4_seed0.pt` by default.

Tests:

```bash
python -m pytest agent_code/rhine/tests -q
```

Evaluation of the final checkpoint in 100 games against three `rule_based_agent`s, oscillation breaker on and thinking-time limit disabled, with the settings used for the report. `--reproducible` fixes the opponents' randomness, so with the checkpoint above the command reproduces the report's training seed 0 results exactly. The checkpoint path must be absolute:

```bash
python -m agent_code.rhine.scripts.eval_stage_d_final \
    --checkpoint "$PWD/agent_code/rhine/models/task4_seed0.pt" --group A \
    --out-dir results/rhine_eval --n-rounds 100 \
    --opponents rule_based_agent rule_based_agent rule_based_agent \
    --no-enable-no-bomb-when-cleared --disable-think-time-limit --reproducible
```

Training of the Task 4 agent for one seed (the checkpoint path must be absolute):

```bash
python -m agent_code.rhine.scripts.run_stage_a2_task3 --stage TASK4 --seed 0 \
    --checkpoint-path-override "$PWD/agent_code/rhine/models/my_task4_seed0.pt"
```

The Task 4 runs in the report additionally passed `--disable-periodic-eval` and `--enable-oscillation-breaker` (the latter only affects evaluation), evaluated snapshots in a separate process with `run_task4_async_eval.py` and chose the checkpoint offline with `select_task4_checkpoint.py`.

Use `--help` on any script for its options.

## Report

The full description of the method, the training procedure, the experiments and the results is in [report/rhine_report.pdf](report/rhine_report.pdf).

## Acknowledgements and sources

The game framework is the course framework at https://github.com/ukoethe/bomberman_rl, used at commit `0f55c1d207e586e6fe6db95e64302ed32281e370`. The upstream repository does not state a license, so this repository does not include any framework code. `setup.sh` downloads it from the original repository.

Implementation was AI-assisted. The design decisions, experiments and analysis were directed by the author.

## License

The code in this repository is released under the MIT License (see [LICENSE](LICENSE)). The license does not apply to the upstream framework.
