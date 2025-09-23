import pandas as pd

from pyaptamer.utils import fasta_to_aaseq


def test_huggingface_url_fetch():
    """Integration test: fetch FASTA from a Hugging Face dataset URL."""
    url = "https://huggingface.co/datasets/gcos/HoloRBP4_round8_trimmed/resolve/main/HoloRBP4_round8_trimmed.fasta"

    sequences = fasta_to_aaseq(url)
    assert isinstance(sequences, list)
    assert len(sequences) > 0
    assert all(isinstance(seq, str) and seq for seq in sequences)

    df = fasta_to_aaseq(url, return_df=True)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["id", "sequence"]
    assert all(df["sequence"].str.len() > 0)
