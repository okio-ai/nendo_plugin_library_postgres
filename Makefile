# Define the commands
SETUP_CMD = pip3 install -e
CLEAN_CMD = rm -rf .coverage* && rm -rf .pytest_cache && rm -rf build && rm -rf dist && rm -rf htmlcov && rm -rf site && find . -type d -name __pycache__ | xargs rm -rf
FORMAT_CMD = python3 -m ruff --fix-only src tests  ; black ./src
# TODO introduce dependency security check
# TODO introduce breaking API changes check
CHECK_CMD = python3 -m ruff check src tests
TEST_CMD = python -m coverage run -m pytest tests/*.py && python -m coverage report && python -m coverage html
ORM_CMD = alembic revision -m "Autogenerated model update" --head=base --branch-label=postgres --version-path=alembic/versions
ORM_UPGRADE_CMD = alembic upgrade postgres@head
POETRY_SETUP_CMD = poetry install --all-extras --sync
POETRY_PUBLISH_CMD = poetry build && poetry publish
CHANGELOG_CMD = git-changelog -c angular -t keepachangelog -s build,deps,feat,fix,refactor,docs -i -o CHANGELOG.md -T --bump=auto .
RELEASE_CMD = git add pyproject.toml CHANGELOG.md && git commit -m "chore: prepare release $(version)" && git tag $(version) && git push && git push --tags

.PHONY: setup setup-poetry clean format check test docs orm orm-upgrade changelog publish release

# Default target
all: format check test orm

# Command targets
setup:
	@$(SETUP_CMD)

setup-poetry:
	@$(POETRY_SETUP_CMD)

clean:
	@$(CLEAN_CMD)

format:
	@$(FORMAT_CMD)

check:
	@$(CHECK_CMD)

test:
	@$(TEST_CMD)

orm:
	@if git diff --name-only | grep -q 'src/nendo/library/model.py'; then \
    	echo "Detected changes in db/model.py. Creating alembic auto-migration..."; \
		$(ORM_CMD); \
	else \
		echo "No changes to model. Skipping alembic auto-migration."; \
		exit 0; \
	fi

orm-upgrade:
	@$(ORM_UPGRADE_CMD)

changelog:
	@$(CHANGELOG_CMD)

release:
	@if [ -z "$(version)" ]; then \
		echo "Please specify a version number (e.g., make release version=0.1.1)"; \
		exit 1; \
	fi
	@$(RELEASE_CMD)

publish:
	@$(POETRY_PUBLISH_CMD)