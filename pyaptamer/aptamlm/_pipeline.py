"""
AptaMLM pipeline for protein-conditioned aptamer generation.
"""

__author__ = ["NoorMajdoub"]
__all__ = ["AptaMLMPipeline"]


import torch
from torch import Tensor
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from pyaptamer import logger
from pyaptamer.aptamlm._model import AptaMLM


class AptaMLMPipeline:
    """AptaMLM pipeline for protein-conditioned aptamer generation.

    Uses a dual-encoder architecture inspired by BAnG and PepMLM
    to generate DNA/RNA aptamer sequences conditioned on a target protein
    sequence. A frozen ESM2 protein encoder and a frozen NucleotideTransformer
    aptamer encoder are bridged by learned cross-attention layers. At
    inference time, all aptamer positions are initialised as ``[MASK]``
    tokens and decoded given the protein context (Decoding strategy still not fixed).

    Parameters
    ----------
    model : AptaMLM
        An initialised ``AptaMLM`` model instance.
    device : torch.device
        Device on which to run inference.
    esm2_model_name : str, optional, default="facebook/esm2_t33_650M_UR50D"
        HuggingFace model identifier for the ESM2 protein encoder.
    nt_model_name : str, optional,
        default="InstaDeepAI/nucleotide-transformer-500m-human-ref"

    Attributes
    ----------
    prot_tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for protein sequences (ESM2).
    nuc_tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for nucleotide sequences (NucleotideTransformer).
    prot_encoder : transformers.PreTrainedModel
        Frozen ESM2 encoder.
    nuc_encoder : transformers.PreTrainedModel
        Frozen NucleotideTransformer encoder.


    """

    def __init__(
        self,
        model: AptaMLM,
        device: torch.device,
        esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",
        nt_model_name: str = (
            "InstaDeepAI/nucleotide-transformer-500m-human-ref"
        ),
    ) -> None:
        self.model = model.to(device)
        self.device = device

        logger.info("Loading ESM2 protein encoder from %s...", esm2_model_name)
        self.prot_tokenizer = AutoTokenizer.from_pretrained(esm2_model_name)
        self.prot_encoder = AutoModel.from_pretrained(esm2_model_name).to(device)
        for param in self.prot_encoder.parameters():
            param.requires_grad = False
        self.prot_encoder.eval()

        logger.info(
            "Loading NucleotideTransformer encoder from %s...", nt_model_name
        )
        self.nuc_tokenizer = AutoTokenizer.from_pretrained(nt_model_name)
        self.nuc_encoder = AutoModelForMaskedLM.from_pretrained(nt_model_name).to(
            device
        )
        for param in self.nuc_encoder.parameters():
            param.requires_grad = False
        self.nuc_encoder.eval()

    @torch.no_grad()
    def _encode_protein(self, sequence: str) -> tuple[Tensor, Tensor]:
        """Tokenize and encode a protein sequence with ESM2.

        Parameters
        ----------
        sequence : str
            Amino-acid sequence string.

        Returns
        -------
        hidden : Tensor
            ESM2 last hidden states of shape ``(1, seq_len_prot, 1280)``.
        padding_mask : Tensor
            Boolean padding mask of shape ``(1, seq_len_prot)``.
            ``True`` indicates a padding position.
        """
        inputs = self.prot_tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        outputs = self.prot_encoder(
            **inputs,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        padding_mask = inputs["attention_mask"] == 0
        return hidden, padding_mask

    @torch.no_grad()
    def _encode_aptamer(self, sequence: str) -> tuple[Tensor, Tensor]:
        """Tokenize and encode an aptamer sequence with NucleotideTransformer.

        Parameters
        ----------
        sequence : str
            DNA/RNA nucleotide sequence string.

        Returns
        -------
        hidden : Tensor
            NucleotideTransformer last hidden states of shape
            ``(1, seq_len_nuc, 1280)``.
        padding_mask : Tensor
            Boolean padding mask of shape ``(1, seq_len_nuc)``.
            ``True`` indicates a padding position.
        """
        inputs = self.nuc_tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        outputs = self.nuc_encoder(
            **inputs,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        padding_mask = inputs["attention_mask"] == 0
        return hidden, padding_mask

    # ------------------------------------------------------------------
    # Public API /Main generation function to be calld
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        protein_sequence: str,  # the reference
        aptamer_length: int = 30,
        top_k: int = 1,
    ) -> str:
        """Generate an aptamer sequence conditioned on a target protein.

        All ``aptamer_length`` positions are initialised as ``[MASK]``
        tokens and decoded in a single forward pass.
            .. todo::
                Implement the decoding strategy
               fixed ``aptamer_length`` argument, analogous to the
               autoregressive decoding in BAnG.

        Returns
        -------
        str
            Generated aptamer sequence as a nucleotide string.
        """
        self.model.eval()

        # encode protein
        prot_hidden, prot_padding_mask = self._encode_protein(protein_sequence)

        # build a fully masked aptamer input
        mask_token_id = self.nuc_tokenizer.mask_token_id
        masked_ids = torch.full(
            (1, aptamer_length),
            fill_value=mask_token_id,
            dtype=torch.long,
            device=self.device,
        )

        # encode the masked aptamer
        nuc_inputs = self.nuc_tokenizer.decode(
            masked_ids[0].tolist(), skip_special_tokens=False
        )
        nuc_hidden, nuc_padding_mask = self._encode_aptamer(nuc_inputs)

        # forward pass through the bridge + MLM head
        self.model(
            prot_hidden,
            nuc_hidden,
            prot_padding_mask=prot_padding_mask,
            nuc_padding_mask=nuc_padding_mask,
        )

        # TODO : Decoding strategy  and return the aptamer generated