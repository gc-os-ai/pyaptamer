import torch
from pyaptamer.aptatrans import (
    AptaTrans,
    AptaTransPipeline,
    EncoderPredictorConfig,
)
from pyaptamer.utils import plot_interaction_map

def main():
    print("Setting up minimal AptaTrans pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We use small max_len for testing
    apta_embedding = EncoderPredictorConfig(128, 16, max_len=10)
    prot_embedding = EncoderPredictorConfig(128, 16, max_len=10)
    prot_words = {"DHR": 0.5, "AIQ": 0.5, "AAG": 0.2}
    
    # Pretrained MUST be False for this toy architecture. We force a seed manually instead so it doesn't scramble.
    torch.manual_seed(42)
    model = AptaTrans(apta_embedding, prot_embedding, pretrained=False)
    pipeline = AptaTransPipeline(device, model, prot_words, depth=5, n_iterations=5)
    
    target = "DHRNENIAIQ"
    aptamer = "ACGUA"

    print("Generating Interaction Map...")
    imap = pipeline.get_interaction_map(aptamer, target)
    print(f"Matrix shape: {imap.shape}")

    print("Opening plot window...")
    # This will open a Matplotlib window
    plot_interaction_map(
        imap,
        candidate=aptamer,
        target=target,
        prot_words=pipeline.prot_words,
        show=True,
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
