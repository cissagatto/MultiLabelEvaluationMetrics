# MultiLabel Evaluation Metrics

A Python implementation of various metrics for evaluating multi-label classification models. This package offers a hierarchical organization of metrics, some of which were implemented from scratch, while others are derived from **scikit-learn**.

## How to Cite

```plaintext
@misc{MLEM2024,
  author = {Elaine Cecília Gatto},
  title = {Multilabel Evalation Metrics: a python implementation for multilabel classification and multilabel ranking models evaluation},  
  year = {2024},  
  doi = {10.13140/RG.2.2.25706.73925},
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
import evaluation as eval

# Sample data
    true_labels = pd.DataFrame({
        'L1': [1, 0, 1, 1],
        'L2': [0, 1, 1, 0],
        'L3': [0, 0, 1, 1],
        'L4': [1, 1, 0, 1]
    })
    pred_labels = pd.DataFrame({
        'L1': [1, 0, 0, 1],
        'L2': [0, 1, 0, 1],
        'L3': [0, 0, 1, 0],
        'L4': [1, 0, 0, 1]
    })
    pred_scores = pd.DataFrame({
        'L1': [1.0, 0.5, 0.0, 0.6],
        'L2': [0.3, 0.4, 0.1, 0.9],
        'L3': [0.2, 0.6, 0.7, 0.8],
        'L4': [0.9, 0.5, 0.3, 0.1]
    })

    res_bipartition = eval.multilabel_bipartition_measures(true_labels, pred_labels)   
    res_ranking = eval.multilabel_ranking_measures(true_labels, pred_scores)
    res_curves = eval.multilabel_curves_measures(true_labels, pred_scores)
    res_lp = eval.multilabel_label_problem_measures(true_labels, pred_labels)
    
    print(res_bipartition)
    print(res_ranking)
    print(res_curves)
    print(res_lp)
```

If you want to calculate specific metrics, use the evaluation functions, for example:

```python
import measures as ms

average_precision = ms.mlem_average_precision(true_labels, pred_scores)
precision_atk = ms.mlem_precision_at_k(true_labels, pred_scores)
coverage = coverage_error(true_labels, pred_scores)
iserror = ms.mlem_is_error(true_labels, pred_scores)
margin_loss = ms.mlem_margin_loss(true_labels, pred_scores)       
ranking_error = ms.mlem_ranking_error(true_labels, pred_scores)       
ranking_loss = label_ranking_loss(true_labels, pred_scores)
one_error = ms.mlem_one_error(true_labels, pred_scores)

# Store all metrics in a dictionary
metrics_dict = {
        'average_precision': average_precision,
        'coverage': coverage,
        'is_error': iserror,
        'margin_loss': margin_loss,
        'precision_atk': precision_atk,
        'ranking_error': ranking_error,
        'ranking_loss': ranking_loss    
}

# Converter o dicionário em um DataFrame com colunas "Measure" e "Value"
metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Measure', 'Value'])
print(metrics_df)
   
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
  - `one_error`
  
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
