.PHONY: all setup clean

all: setup
	@echo "Setup finished. You can now run CLSA"

setup:
	@echo "Creating model directories..."
	mkdir -p models/translation models/encoders
	@echo "Cloning translation model..."
	cd models/translation && \
		git clone https://huggingface.co/facebook/m2m100_418M || echo "Already cloned"
	@echo "Cloning encoder models..."
	cd ../encoders && \
		git clone https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest || echo "Already cloned" && \
		git clone https://huggingface.co/unitary/toxic-bert || echo "Already cloned" && \
		git clone https://huggingface.co/j-hartmann/emotion-english-distilroberta-base || echo "Already cloned" && \
		git clone https://huggingface.co/cardiffnlp/twitter-roberta-base-irony || echo "Already cloned" && \
		git clone https://huggingface.co/cointegrated/roberta-base-formality || echo "Already cloned" && \
		git clone https://huggingface.co/IDA-SERICS/PropagandaDetection || echo "Already cloned"
	cd ../..
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

clean:
	rm -rf models