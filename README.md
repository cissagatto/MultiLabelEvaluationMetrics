# MultiLabel Evaluation Metrics

A Python implementation of various metrics for evaluating multi-label classification models. This package offers a hierarchical organization of metrics, some of which were implemented from scratch, while others are derived from **scikit-learn**.

## How to Cite

```plaintext
@misc{MLEM2024,
  author = {Elaine Cecília Gatto},
  title = {Multilabel Evalation Metrics: a python implementatior for multilabel classification and multilabel ranking models evaluation},  
  year = {2024},  
  doi = {},
  url = {https://github.com/cissagatto/MultiLabelEvaluationMetrics}
}
```

## Installation

To install and use the metrics, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/cissagatto/MultiLabelEvaluationMetrics.git
   cd MultiLabelEvaluationMetrics
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The metrics can be directly called to evaluate your multi-label classification models. Here's a sample usage:

```python
from evaluation import get_all_measures_names

# Example of loading all measure names
measures = get_all_measures_names()

# Print all macro-based metrics
print(measures['macro-based'])
```

If you want to calculate specific metrics, use the evaluation functions, for example:

```python
import evaluation as eval

true_labels = [...]
pred_labels = [...]

# Get label-problem specific metrics
result = eval.multilabel_label_problem_measures(true_labels, pred_labels)
print(result)
```

## Features

- **Bipartition Metrics**:
  - Example-based: `accuracy`, `hamming-loss`, `zero-one-loss`
  - Label-based: macro/micro/weighted metrics like `precision`, `recall`, `f1-score`, and `roc-auc`
  
- **Ranking Metrics** (manually implemented):
  - `average-precision`
  - `margin-loss`
  - `one-error`
  - `ranking-loss`
  - `ranking-error`
  
- **Label Problem Metrics** (manually implemented):
  - `clp` (Constant Label Problem)
  - `mlp` (Missing Label Problem)
  - `wlp` (Wrong Label Problem)

- **Scores**:
  - `auprc` (Area Under Precision-Recall Curve)
  - `roc` (Receiver Operating Characteristic)

## 📚 **Contributing**

If you'd like to contribute to the development of this project, feel free to open an issue or submit a pull request. Contributions that enhance functionality and performance are always welcome!

## 📧 **Contact**

For any questions or support, please contact:
- **Prof. Elaine Cecilia Gatto** (elainececiliagatto@gmail.com)

## Acknowledgment

- This study was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code 001.
- This study was partly financed by the Conselho Nacional de Desenvolvimento Científico e Tecnológico - Brasil (CNPQ) - Process number 200371/2022-3.
- The authors also thank the Brazilian research agency FAPESP for financial support.

## Links

| [Site](https://sites.google.com/view/professor-cissa-gatto) | [Post-Graduate Program in Computer Science](http://ppgcc.dc.ufscar.br/pt-br) | [Computer Department](https://site.dc.ufscar.br/) | [Biomal](http://www.biomal.ufscar.br/) | [CNPQ](https://www.gov.br/cnpq/pt-br) | [Ku Leuven](https://kulak.kuleuven.be/) | [Embarcados](https://www.embarcados.com.br/author/cissa/) | [Read Prensa](https://prensa.li/@cissa.gatto/) | [LinkedIn Company](https://www.linkedin.com/company/27241216) | [LinkedIn Profile](https://www.linkedin.com/in/elainececiliagatto/) | [Instagram](https://www.instagram.com/cissagatto) | [Facebook](https://www.facebook.com/cissagatto) | [Twitter](https://twitter.com/cissagatto) | [Twitch](https://www.twitch.tv/cissagatto) | [YouTube](https://www.youtube.com/CissaGatto) |

---

Happy coding! 🚀
