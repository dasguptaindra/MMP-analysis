🧪 Matched Molecular Pair (MMP) Analysis Tool

A Streamlit-based interactive application for performing Matched Molecular Pair (MMP) analysis to identify structural transformations that influence compound potency (pIC50).
The tool is designed to faithfully reproduce classical MMP logic, with a strong emphasis on statistical robustness, interpretability, and medicinal chemistry relevance.

📌 Key Features

🔬 MMP logic implementation

Pairs generated only when ≥3 compounds share the same core

Canonical SMILES-based ordering for reproducibility

🧩 Automated molecular fragmentation

Single-cut fragmentation strategy

Core–R-group decomposition

📈 Transform effect analysis

ΔpIC50 calculation for each matched pair

Mean ΔpIC50 and distribution per transformation

🧠 Medicinal chemistry interpretability

Reaction SMARTS visualization

Positive and negative transformation ranking

🖼 Structure visualization

Reaction schemes

Compound examples (before/after)

📊 Interactive analytics

Strip plots for ΔpIC50 distributions

Expandable tables and molecule grids

💾 Export options

CSV and Excel outputs for downstream analysis

🧬 Scientific Background

Matched Molecular Pair Analysis (MMPA) is a ligand-based technique used in medicinal chemistry to identify the impact of small, well-defined chemical changes on biological activity.

This implementation follows the principles described in:

Hussain & Rea, J. Chem. Inf. Model., 2010

Dossetter et al., Drug Discovery Today, 2013

Tyrchan & Evertsson, CSBJ, 2017

📂 Input File Requirements

Upload a CSV file with the following columns:

Column	Required	Description
SMILES	✅	Molecular structure in SMILES format
pIC50	✅	Potency value (–log₁₀ IC50)
Name		Compound identifier (recommended)
Example CSV
SMILES,Name,pIC50
CCOc1ccc(C(=O)N2CCNCC2)cc1,Compound_1,6.3
CCOc1ccc(C(=O)N3CCC(CC3)O)cc1,Compound_2,7.1
CCOc1ccc(C(=O)N4CCOCC4)cc1,Compound_3,5.8

⚙️ Application Workflow

Upload dataset

Molecule preprocessing

Optional sanitization

Optional kekulization

Fragmentation

Single-bond cuts

Largest fragment retained as core

Pair generation

Same core, different R-groups

ΔpIC50 calculation

Transform frequency filtering

Visualization & export

⚠️ Limitations

Single-cut fragmentation only

No stereochemistry handling

Activity assumed comparable across assays

Not suitable for covalent or metal-binding ligands (without modification)

📚 References

Hussain, J., Rea, C. J. Chem. Inf. Model., 2010

Dossetter, A. G., et al. Drug Discovery Today, 2013

Wassermann, A. M., et al. Drug Dev. Res., 2012

Tyrchan, C., Evertsson, E. CSBJ, 2017

📜 License

For academic and research use only.
Please validate all computational insights with experimental data.

👨‍🔬 Author Notes

Designed for computational medicinal chemists, QSAR researchers, and drug discovery scientists who require:

Transparent MMP logic

Reproducible results

Interpretable chemical insights
