# Guidelines for Declaring the Use of Large Language Models in Project Reports

**Course:** Qauntum Computing and Quantum machine Learning (FYS5419/9419 University of Oslo)**  
**Applies to:** Last project and its  report and submitted code

---

## 1. Preamble

Large language models (LLMs) such as ChatGPT, Claude, GitHub Copilot, Gemini, and similar tools are
permitted and even encouraged in this course. They are powerful aids for writing code, exploring
unfamiliar concepts, drafting text, and correcting language.

However, **using an LLM does not reduce your intellectual responsibility for what you submit.**
Every result, derivation, code block, and argument in your report must be something you understand,
can defend, and have verified. An LLM can hallucinate plausible-sounding but wrong equations,
incorrect references, and subtly broken code. Catching and correcting such errors is itself part
of the learning process and is your responsibility.

The declaration requirements below are designed to be:

- **lightweight** — a short, structured appendix, not an essay,
- **specific** — tied to concrete tasks, not vague,
- **honest** — a genuine record of how the tool shaped the work.

---

## 2. General Principles

**Principle 1 — Transparency.**  
Any significant use of an LLM must be declared. "Significant" means the LLM contributed content,
logic, or structure that appears in the final submission, not merely that you consulted it for
background reading you then independently summarised.

**Principle 2 — Ownership.**  
You are the author of your report. LLMs are tools, like a compiler, a spell-checker, or a
textbook. Declaring their use is analogous to citing a reference — it does not diminish your
authorship; concealing it does.

**Principle 3 — Verifiability.**  
Any code generated or substantially assisted by an LLM must be tested, understood, and commented
by you. Any text written or heavily revised by an LLM must be read, understood, and endorsed by
you. If you cannot explain a piece of code or a paragraph in your own words during a potential
oral follow-up, it should not be in the report.

**Principle 4 — Scientific integrity.**  
Do not ask an LLM to fabricate results, invent references, or make up numerical values. If you
use an LLM to help find references, verify every citation against the actual source before
including it.

---

## 3. What Must Be Declared

The table below maps task types to the required level of declaration.

| Task | Declaration required? | Where to declare |
|---|---|---|
| Spell-checking / grammar correction | Yes (brief) | Appendix, one line |
| Rephrasing for clarity (minor) | Yes (brief) | Appendix, one line |
| Structural editing (reorganising sections, rewriting paragraphs) | Yes (detailed) | Appendix, per section |
| Drafting an entire section from a prompt | Yes (detailed) | Appendix + inline note |
| Generating a code skeleton or template | Yes | Code comment + Appendix |
| Debugging code (LLM identified the bug) | Yes | Code comment |
| Generating a complete function or class | Yes | Docstring + Appendix |
| Explaining a concept you then summarised in your own words | No | — |
| Literature search (you verified all citations) | No | — |
| Brainstorming project ideas (you chose and developed them) | No | — |

---

## 4. How to Declare: The LLM Appendix

Add a dedicated appendix section titled **"Appendix: Use of AI/LLM Tools"** at the end of your
report (before the reference list, or after — be consistent). The appendix must contain the
following subsections where applicable.

### 4.1 Tools Used

List every LLM tool used in the project.

**Template:**

```
Tools used: [Tool name, version or access date, access method]

Example:
- ChatGPT (GPT-4o, accessed via chat.openai.com, January–May 2026)
- Claude (Claude Sonnet, accessed via claude.ai, February 2026)
- GitHub Copilot (integrated in VS Code, throughout the project)
```

### 4.2 Text Writing and Editing

For each section of the report, state the role the LLM played using one of the four
**contribution levels** defined below.

**Contribution levels:**

| Level | Label | Meaning |
|---|---|---|
| 0 | None | No LLM assistance |
| 1 | Language only | Spell-check, grammar, minor phrasing |
| 2 | Editorial | LLM rewrote or restructured sentences/paragraphs; you supplied all ideas and checked the result |
| 3 | Generative | LLM drafted substantial text from your prompts; you revised, verified, and took ownership |

**Template:**

```
Section              | LLM contribution level | Notes
---------------------|------------------------|------------------------------
Abstract             | 1                      | Grammar check only
Introduction         | 2                      | Rephrased two paragraphs for clarity
Theory / Methods     | 0                      | Written independently
Results              | 0                      | Written independently
Discussion           | 2                      | LLM suggested reorganisation of Sec. 4.2
Conclusion           | 1                      | Minor language corrections
```

If any section is Level 3, add a sentence describing what prompt(s) you used and what changes
you made to the LLM output.

**Example (Level 3 note):**

> *Section 2.3 (Variational Quantum Eigensolver):* I prompted Claude with "Explain the
>  VQE for a student at graduate level, assuming the reader knows the variational principle." The
> output was used as a starting point; I rewrote the notation to match the rest of the report,
> corrected a sign error in equation (7), and added the connection to our specific code
> architecture.

### 4.3 Code Generation and Assistance

For each code file or notebook submitted, state the LLM's role using the levels below.

**Code contribution levels:**

| Level | Label | Meaning |
|---|---|---|
| 0 | None | No LLM assistance |
| 1 | Debugging | LLM helped identify a bug; fix implemented by you |
| 2 | Snippet | LLM provided a function, loop, or short block; you integrated and tested it |
| 3 | Skeleton | LLM generated the overall structure of a script/class; you filled in domain-specific logic |
| 4 | Substantial | LLM wrote the majority of a file; you adapted, tested, and commented it |

**Template:**

```
File / notebook           | LLM level | Description
--------------------------|-----------|----------------------------------------------
train.py                  | 2         | LLM provided the shot/measurement setup;
                          |           | tested and adapted for our NTK experiment
qaoa.ipynb                | 3         | LLM generated class skeleton for QAOA;
                          |           | Optimization architecture designed by us
utils/ntk_compute.py      | 0         | Written independently
results/plot_figures.py   | 1         | LLM debugged an indexing error in the
                          |           | eigenvalue sorting routine
```

Additionally, **every LLM-generated or LLM-assisted function must carry an inline comment** in
the source code itself (see Section 5).

---

## 5. In-Code Documentation Standards

### 5.1 Function-level docstring tag

For any function where the LLM contributed at Level 2 or above, add a `LLM-assisted` tag to
the docstring, stating the tool and a brief description of what it did.

**Python example:**

```python
def compute_nat_grad(model, X, device="cpu"):
    """
    Compute the Natural Gradient 
    for a given model and input data X.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network (should be in eval mode).
    X : torch.Tensor, shape (N, d)
        Input data matrix.
    device : str
        Torch device string.

    Returns
    -------
    K : torch.Tensor, shape (N, N)
        Natural gradient.

    LLM-assisted
    ------------
    Tool: GitHub Copilot (March 2026)
    Role: Generated the Jacobian accumulation loop using torch.autograd.functional.jacobian.
    Modifications: Added batching over inputs to avoid OOM on GPU; verified output against
    finite-difference approximation on a two-layer network.
    """
    ...
```

**Fortran example:**

```fortran
! LLM-assisted: Claude (Feb 2026) suggested the Natural Gradient  kernel
! evaluation structure. Loop bounds and array indexing corrected by author.
! Verified against Python reference implementation.
SUBROUTINE compute_natgrad_kernel(X1, X2, depth, sigma_w, sigma_b, K_out)
    ...
END SUBROUTINE
```

**C++ example:**

```cpp
/**
 * Computes the Nat gradient covariance kernel Sigma^{(L)} recursively.
 *
 * LLM-assisted: ChatGPT (GPT-4o, April 2026) generated the base
 * template for the recursive lambda evaluation. The Gaussian quadrature
 * integration was replaced by the author with a Gauss-Hermite scheme
 * appropriate for the activation function used.
 */
double compute_sigma(double x, double xp, int depth,
                     double sigma_w, double sigma_b);
```

### 5.2 Inline comments for specific LLM-generated blocks

For blocks shorter than a full function (e.g., a data-loading one-liner, a regex, a plotting
snippet), add a short inline comment:

```python
# LLM (Copilot): suggested this vectorised einsum formulation; verified correctness.
K = torch.einsum('ij,kj->ik', J, J)
```

### 5.3 Jupyter notebook cells

In Jupyter notebooks, add a Markdown cell immediately above any LLM-assisted code cell with a
brief declaration:

```markdown
> **LLM note:** The following cell was generated with Claude assistance (Level 3 skeleton).
> The diffusion forward process loop was written by the LLM; the noise schedule and the
> DDPM sampling loop were implemented independently.
```

---

## 6. A Minimal Complete Example

Below is a self-contained example of how the LLM appendix might look in a real report.

---

### Appendix A: Use of AI/LLM Tools

**A.1 Tools used**

- Claude Sonnet (claude.ai, accessed February–May 2026)
- GitHub Copilot (VS Code extension, throughout the project)

**A.2 Text contributions**

| Section | Level | Notes |
|---|---|---|
| Abstract | 1 | Grammar check |
| 1. Introduction | 0 | — |
| 2. Theory | 2 | Section 2.4 (NTK recursion): LLM reformatted equations for readability |
| 3. Methods | 0 | — |
| 4. Results | 0 | — |
| 5. Discussion | 2 | LLM suggested restructuring paragraph order in Sec. 5.2 |
| 6. Conclusion | 1 | Minor grammar corrections |

*Level 2 note, Section 2.4:* I gave Claude the recursion from XX et al. (2018) and asked it
to rewrite it in notation consistent with Section 2.1 of this report. I checked every equation
against the original paper.

**A.3 Code contributions**

| File | Level | Description |
|---|---|---|
| `src/ntk.py` | 3 | Copilot generated Jacobian loop; Gradient recursion and verification code written independently |
| `src/train_vae.py` | 2 | Copilot autocompleted the quantum annealing scheduler; adapted to our $\beta$-schedule |
| `notebooks/results.ipynb` | 1 | Copilot debugged a matplotlib axis-label bug |
| `src/utils.py` | 0 | Fully independent |

---

## 7. Frequently Asked Questions

**Q: I used an LLM to explain a concept in a textbook, then wrote the theory section myself. Do I need to declare this?**  
No. Using an LLM as an interactive textbook or tutor, when the final writing is entirely your
own, is not different from reading Wikipedia or a review article. No declaration needed.

**Q: Copilot autocompletes almost everything I type. Must I declare every autocomplete?**  
No — single-line autocompletes of boilerplate (import statements, obvious variable names) need
not be declared individually. Declare at Level 1 ("Copilot used throughout for autocompletion")
in Section A.1. Any multi-line or logically non-trivial suggestion you accepted should be noted
at the file level in the code table.

**Q: What if I used the LLM heavily and the contribution is genuinely hard to separate?**  
Be honest and say so. Write something like: "Substantial LLM assistance throughout; all code
was tested, all derivations verified, all text read and understood by the author." This is
acceptable — concealment is not.

**Q: Can I paste in LLM-generated code if I do not understand it?**  
No. Understanding what you submit is a core requirement. If you cannot explain a piece of code,
do not submit it. Use the LLM output as a learning resource, then rewrite it yourself.

**Q: What happens if I do not declare LLM use?**  
Undeclared LLM use that is later identified will be treated as a violation of the University
of Oslo's academic integrity policy, in the same way as plagiarism from any other source.

**Q: Does heavy LLM use lower my grade?**  
No, provided you demonstrate understanding. The grade reflects the quality of the science,
the correctness of the implementation, the clarity of the analysis, and your ability to
discuss the work. How you produced it matters less than what you produced and what you
understand.

---

## 8. Summary Checklist

Before submitting your report, verify the following:

- [ ] Appendix section "Use of AI/LLM Tools" is present
- [ ] All LLM tools used are listed with name and approximate dates
- [ ] Every section of the report has a declared contribution level (0–3)
- [ ] Level 3 sections have a brief prompt/modification note
- [ ] All LLM-assisted functions have a `LLM-assisted` tag in their docstring
- [ ] LLM-generated code blocks have inline comments
- [ ] Jupyter notebooks have Markdown declaration cells above assisted code cells
- [ ] All cited references have been verified against the actual source
- [ ] You can explain every equation, result, and code block in your own words

---

