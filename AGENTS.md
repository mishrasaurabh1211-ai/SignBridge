# Development Conventions
1. One implementation per concept. The preprocessing used in training is the same function used in live inference. Never fork it.
2. Data and feature code are pure functions over numpy arrays. torch is used only inside models/ and training.
3. Every tunable lives in configs/*.yaml and is read through config.py. No magic numbers in code.
4. Every module ships with tests. Tests run without the dataset (synthetic fixtures) unless marked @pytest.mark.dataset.
5. No notebooks in the repo. Analysis is a script under scripts/ that writes to artifacts/.
6. Report only numbers produced by running code in this session. Paste the command and its output.
7. Interfaces defined in an earlier phase are frozen. Changing one means updating every caller and test in the same change and saying so.
8. Do not add a dependency that is not in pyproject without stating why in the PR description.
9. Finish the whole phase. If something is blocked, deliver everything else and say exactly what is missing.
