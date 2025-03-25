# Contributing to `blase`

Thank you for your interest in contributing to `blase`, together we will help people and their machines learn better! Contributions are welcome in the form of bug reports, feature requests, code contributions, and documentation improvements.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a new branch for your contribution:
   ```sh
   git checkout -b feature-or-bugfix-name
   ```
3. Make your changes following the guidelines below.
4. Commit with a descriptive message:
   ```sh
   git commit -m "Brief description of the change"
   ```
5. Push your branch to GitHub:
   ```sh
   git push origin feature-or-bugfix-name
   ```
6. Open a pull request (PR) against the 'main' branch.

## Reporting Issues
If you encounter a bug or have a feature request, open an issue with the following details:
- A clear title and description of the problem.
- Steps to reproduce the issue, if applicable.
- Expected behavior vs. actual behavior.
- Relevant error messages, logs, or screenshots.
Bug reports should include as much detail as possible to help diagnose the issue.

## Code Contributions
### 1. Testing
All contributions should include tests to verify correctness:
- Unit tests should be placed in the tests/ directory.
- Run tests before submitting a PR:
   ```sh
   pytest
   ```
### 2. Documentation
All major contributions should be accompanied by documentation updates:
- If adding a new feature, update relevant sections in 'README.md'.
- If modifying core functionality, ensure function docstrings are clear and concise.

## Pull Request Guidelines
- Each pull request should address a single issue or feature.
- Include a clear description of what the PR does.
- Reference related issues in the PR description using 'Closes #issue-number'.
- Ensure tests pass before submitting.
- Keep commits focused and concise.

## License
By contributing to 'blase', you agree that your contributions will be licensed under the **BSD 3-Clause License**.

For any further questions, feel free to open an issue or reach out to the maintainers.
