PYTHON=python

train:
	@if [ -z "$(FROM_DATE)" ]; then \
		echo "Error: FROM_DATE is not set. Use: make train FROM_DATE=YYYY-MM-DD"; \
		exit 1; \
	fi
	$(PYTHON) ./commands/train_model.py --from_date=$(FROM_DATE)


run:
	$(PYTHON) ./main.py