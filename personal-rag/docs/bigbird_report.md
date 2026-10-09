# Task 0: BigBird Transformer

## 1. Motivation

Standard Transformers such as BERT use full self-attention: every token attends to every other token. For a sequence of length `n` this costs **O(n²)** time and memory, because the attention matrix is n × n. In practice BERT is limited to **512 tokens**. Long documents (books, legal contracts, scientific papers, genomes) must then be truncated or split, which loses cross-chunk context.

**BigBird** (Zaheer et al., Google Research, NeurIPS 2020, "Big Bird: Transformers for Longer Sequences") replaces full attention with a **sparse attention** pattern whose cost is **linear, O(n)**, while keeping the theoretical strengths of full attention. It handles sequences of up to **4096 tokens**, about 8× more than BERT, on the same hardware.

## 2. Architecture

BigBird keeps the standard Transformer encoder (multi-head attention, feed-forward layers, residuals, layer norm). The only change is **which query-key pairs are computed** in the attention layer. The attention graph combines three components:

| Component | What it does | Role |
|---|---|---|
| **Window (local) attention** | Each token attends to `w` neighbours (sliding window) | Captures local context, as in CNN-style locality |
| **Global tokens** | `g` special tokens attend to all tokens and all tokens attend to them | Information hubs that give short paths between any two tokens |
| **Random attention** | Each token attends to `r` randomly chosen tokens | Acts like a random graph: very short average path length, fast information mixing |

Each token therefore attends to about `w + r + g` tokens, a constant that does not grow with `n`.

### Global token variants
- **BigBird-ITC (Internal Transformer Construction):** some existing tokens (e.g. `[CLS]`) are made global.
- **BigBird-ETC (Extended Transformer Construction):** extra auxiliary tokens are added and made global. This gives more capacity, and works well for QA and structured tasks.

## 3. Block-sparse implementation

Sparse attention at the single-token level is inefficient on GPUs/TPUs because of irregular memory access. BigBird therefore **groups tokens into blocks** (e.g. block size 64) and applies the sparse pattern at the block level:

- **Sliding blocks:** each query block attends to itself and its neighbouring blocks (e.g. 3 blocks).
- **Global blocks:** the first and last blocks (or ETC blocks) attend to and are attended by everything.
- **Random blocks:** each query block attends to `r` random key blocks (e.g. 3).

Block-level attention can be computed with dense matrix multiplications on small blocks (gather, then batched matmul), which uses hardware efficiently. In HuggingFace, `attention_type="block_sparse"` is used for long inputs and falls back to `original_full` when the sequence is too short for the sparse pattern to help.

## 4. How it fixes the quadratic cost

Full attention: n × n scores. BigBird: n × (w + r + g) scores, with `w + r + g` fixed. So memory and time scale **linearly** with length.

Why it still works well (theory from the paper):
- The attention pattern is a graph with small diameter (random edges + global hubs), so any token can influence any other within a few layers.
- BigBird is a **universal approximator** of sequence-to-sequence functions.
- It is **Turing complete**, as full-attention Transformers are.
- The paper also shows that some tasks cannot be done by sparse attention as cheaply as by full attention (a separation result), so it is not strictly equivalent.

## 5. Complexity comparison

| Model | Attention pattern | Complexity per layer | Typical max length |
|---|---|---|---|
| **BERT** | Full | **O(n²·d)** | 512 |
| **Longformer** | Sliding window + task-specific global tokens | **O(n·w·d)** (linear) | 4096 (up to 16K in variants) |
| **BigBird** | Sliding window + global + **random** | **O(n·(w+r+g)·d)** (linear) | 4096 |

Longformer vs BigBird: both are linear. BigBird adds **random attention** and has the formal expressiveness guarantees. Longformer is simpler and relies on window + global tokens chosen per task. In practice their quality is similar, and BigBird often does better on tasks that need distant token interaction (e.g. long-document QA).

## 6. Suitable applications for long documents

- **Long-document question answering** (HotpotQA, Natural Questions, TriviaQA): evidence is spread across thousands of tokens.
- **Abstractive summarization** of long articles, papers and reports (BigBird-Pegasus on arXiv / PubMed).
- **Long-document classification** (legal, medical records, patents).
- **Genomics:** DNA sequences are very long; the paper reports gains on promoter-region and chromatin-profile prediction.
- **Retrieval/RAG pipelines** that need larger passages or whole-document encoders.

## 7. Advantages

- Linear time and memory, so longer inputs on the same hardware.
- Preserves global context via global tokens and random edges.
- Theoretical guarantees (universal approximation, Turing completeness).
- Can be initialized from pretrained BERT/RoBERTa-style weights and extended.
- State-of-the-art results at the time on long-context QA and summarization.

## 8. Limitations

- More complex to implement and optimize than full attention; depends on block-sparse kernels.
- Random attention is hard to optimize on hardware, and results vary by seed/pattern.
- Large constant factors: for short sequences (below about 1024 tokens) full attention is as fast or faster, and BigBird gives no benefit.
- Sparse attention can lose some precision on tasks that need dense token-to-token interactions.
- Still bounded at 4096 tokens in most checkpoints, which is not unlimited.
- Fewer pretrained checkpoints and weaker ecosystem support than BERT.

## 9. Performance characteristics (summary)

- Reduces memory enough to run 4096-token sequences where BERT-style models would run out of memory.
- Beat RoBERTa and other baselines on long-context QA and summarization benchmarks in the paper.
- Training and inference cost grows linearly with length, versus quadratically for BERT.
- Best trade-off when inputs are long (>1K tokens); for short text, BERT/RoBERTa remain the simpler choice.

## References
- Zaheer et al., *Big Bird: Transformers for Longer Sequences*, NeurIPS 2020 (arXiv:2007.14062).
- Beltagy et al., *Longformer: The Long-Document Transformer*, 2020 (arXiv:2004.05150).
- Devlin et al., *BERT*, 2018 (arXiv:1810.04805).
- HuggingFace blog: *Understanding BigBird's Block Sparse Attention*.
