# Workspace bash init for the lyapunov Conda env.
# Loaded via `bash --rcfile` from .vscode/settings.json so we get a clean
# interactive shell (no system-wide startup files that call `module`).

source /vast/parcc/spack/sw/apps/linux-zen4/miniconda3-25.5.1-dypgjukzsoebnoyhfrbijohg2uizpmyx/etc/profile.d/conda.sh
conda activate /vast/projects/kostas/geometric-learning/wenliao/conda_envs/lyapunov

export PS1='\[\e[01;32m\]\u@\h\[\e[00m\]:\[\e[01;34m\]\w\[\e[00m\]\$ '

alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
