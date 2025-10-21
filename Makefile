.PHONY: fmt lint type test run

fmt:
\truff format .

lint:
\truff check .

type:
\tmypy app

test:
\tpytest -q

run:
\tuvicorn app.api:app --reload
