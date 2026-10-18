# RHINE: a curriculum-trained PPO agent for Bomberman

Author: Qin Yan

RHINE is a reinforcement learning agent for the four-player Bomberman framework of the course *Machine Learning Essentials* at Heidelberg University (summer semester 2026). It was the author's individually developed part of a group project. This repository contains only the author's own work.

## Method

- **PPO actor-critic.** A multilayer perceptron with a shared trunk of two 64-unit tanh layers, a policy head over the six actions and a value head, trained with the clipped PPO objective and generalized advantage estimation.
- **31 engineered features.** Computed from breadth-first-search distances, blast coverage and time-dependent danger. No feature encodes a recommended action.
- **Rule-based action mask.** Removes illegal actions and actions that lead to a known, unavoidable death; the policy chooses among the remaining ones, in training and in play.
- **Oscillation breaker.** An evaluation-only refinement of the mask that stops the agent from stepping back and forth between two tiles. It is never active during training.
- **Four-task curriculum.** Coins on an empty board, crates and bombs, passive and then coin-collecting opponents, and finally three `rule_based_agent`s. Every task and stage was trained from a random initialization; no weights were transferred.

## Results

Task 4 against three `rule_based_agent`s on the `classic` scenario: 100 games per training seed (evaluation seed 1000), mean ± sample standard deviation over three training seeds, oscillation breaker on. The baseline is the agent from the last Task 3 stage (three `coin_collector_agent`s), which was never trained against `rule_based_agent`.

| Agent | Score | Kills | Self-kill rate | Killed-by-opponent rate | Oscillation fraction |
|---|---|---|---|---|---|
| Task 4 agent | 6.33 ± 0.76 | 0.58 ± 0.19 | 0.113 ± 0.025 | 0.130 ± 0.026 | 0.007 ± 0.006 |
| Stage D agent (baseline) | 5.00 ± 0.54 | 0.34 ± 0.07 | 0.153 ± 0.104 | 0.107 ± 0.006 | 0.007 ± 0.012 |

In the same games, the best of the three opponents averaged 5.04 points per game, and RHINE finished alone in first place in 53% of games (63% including ties). Under a second evaluation seed (2000), RHINE scored 6.11 against 4.84 and finished alone in first place in 52% of games.

All evaluations in the report were run with the framework's per-step thinking-time limit disabled, so that load fluctuations on the evaluation machine could not force WAIT actions. The decision time of the final agent was measured separately in 20 games: the mean `act()` time was 15.8 ms and the maximum 125.2 ms, well below the 0.5 s limit.

## Key findings

- Oscillation turned out to be a limitation of the Markovian state rather than of the reward. The evaluation-time breaker removed it at no cost in score, although it is a rule, not a learned skill.
- Reward shaping destabilized learning on the dense coin board but was indispensable once crates made rewards sparse.
- The Task 4 agent scored significantly higher than the Stage D agent without measurably fewer deaths; the gain came from more kills.
- Most remaining deaths come from losing a same-step contest for a tile, which the agent cannot control.
- The score differs between training seeds by more than it differs between the Task 4 agent and the Stage D agent.

## Repository layout

```
agent_code/
  rhine/            Agent code, the final checkpoint (models/), training and evaluation scripts (scripts/) and tests (tests/)
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

Requires Python 3.12 on Linux, macOS or WSL, and git.

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

Evaluation of the final checkpoint in 100 games against three `rule_based_agent`s, oscillation breaker on and thinking-time limit disabled as in the report (the checkpoint path must be absolute):

```bash
python -m agent_code.rhine.scripts.eval_stage_d_final \
    --checkpoint "$PWD/agent_code/rhine/models/task4_seed0.pt" --group A \
    --out-dir results/rhine_eval --n-rounds 100 \
    --opponents rule_based_agent rule_based_agent rule_based_agent \
    --no-enable-no-bomb-when-cleared --disable-think-time-limit
```

Training of the Task 4 agent for one seed (the checkpoint path must be absolute):

```bash
python -m agent_code.rhine.scripts.run_stage_a2_task3 --stage TASK4 --seed 0 \
    --checkpoint-path-override "$PWD/agent_code/rhine/models/my_task4_seed0.pt"
```

Use `--help` on any script for its options.

## Report

The full description of the method, the training procedure, the experiments and the results is in [report/rhine_report.pdf](report/rhine_report.pdf).

## Acknowledgements and sources

The game framework is the course framework at https://github.com/ukoethe/bomberman_rl, used at commit `0f55c1d207e586e6fe6db95e64302ed32281e370`. The upstream repository does not state a license, so this repository does not include any framework code; `setup.sh` downloads it from the original repository.

Implementation was AI-assisted; the design decisions, experiments and analysis were directed by the author.

## License

The code in this repository is released under the MIT License (see [LICENSE](LICENSE)). The license does not apply to the upstream framework.
