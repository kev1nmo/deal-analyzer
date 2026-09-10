# Phase 5 findings: text feature from listing descriptions

## 1. Approach

- Text: listing title + description, with **every number replaced by `[num]`**. 15% of descriptions state the exact rent, and many give square footage or prices, so leaving numbers in would leak the answer.
- Model: `sentence-transformers/all-MiniLM-L6-v2` (22M parameters), fine-tuned in PyTorch with a mean-pooling + linear head to predict log(rent per sq ft). Huber loss, AdamW (encoder lr 3e-5, head lr 1e-3), linear warmup and decay, 3 epochs, max 160 tokens, Apple GPU (MPS).
- Comparison: the same model frozen (no training). Its embeddings go into a ridge regression with the penalty picked by internal CV.
- No leakage: a training listing's text score comes from a model fit on the *other* 4 spatial folds. Test scores come from a model fit on all training data. The score is one extra column for XGBoost. XGBoost keeps its Phase 4 settings, so the ablation changes only the feature.

## 2. How well does text alone predict rent per sq ft?

| set                 | model                 |   mae_log_psf |   spearman |
|:--------------------|:----------------------|--------------:|-----------:|
| train (out-of-fold) | predict the mean      |         0.264 |    nan     |
| train (out-of-fold) | frozen MiniLM + ridge |         0.195 |      0.650 |
| train (out-of-fold) | fine-tuned MiniLM     |         0.171 |      0.736 |
| test                | predict the mean      |         0.254 |    nan     |
| test                | frozen MiniLM + ridge |         0.210 |      0.386 |
| test                | fine-tuned MiniLM     |         0.216 |      0.469 |

`mae_log_psf` is on the log scale: 0.17 means the typical miss is about 17% of rent per sq ft.

## 3. Ablation: rent model with vs without the text feature

| variant                   |   cv_mae |   test_mae |   test_mape |   test_gain_vs_no_text_pct |   gain_ci_low |   gain_ci_high |
|:--------------------------|---------:|-----------:|------------:|---------------------------:|--------------:|---------------:|
| no text (Phase 4)         |    233.3 |      155.0 |        12.4 |                        0.0 |           0.0 |            0.0 |
| + frozen MiniLM score     |    213.4 |      158.3 |        12.6 |                       -2.2 |          -5.1 |            0.6 |
| + fine-tuned MiniLM score |    206.5 |      156.8 |        12.2 |                       -1.2 |          -4.8 |            2.2 |

`test_gain_vs_no_text_pct`: % MAE reduction vs the Phase 4 model, with a 95% paired bootstrap CI.

**Result:** the fine-tuned text score changes test MAE from $155 to $157 (-1.2%, 95% CI -4.8% to 2.2%).

Caveat on the CV column: a training listing's text score came from a model that saw the other folds, including whichever fold is the XGBoost validation fold at that moment. That is a small second-order leak, so the CV gain is slightly optimistic. The test column has no such leak and is the headline.

## 4. Fine-tuning loss (normalized Huber loss per epoch)

| fit               |   epoch_1 |   epoch_2 |   epoch_3 |
|:------------------|----------:|----------:|----------:|
| fold 0 held out   |     0.336 |     0.190 |     0.152 |
| fold 1 held out   |     0.342 |     0.196 |     0.152 |
| fold 2 held out   |     0.322 |     0.173 |     0.134 |
| fold 3 held out   |     0.332 |     0.170 |     0.134 |
| fold 4 held out   |     0.330 |     0.181 |     0.142 |
| all training data |     0.312 |     0.170 |     0.136 |
## 5. Interpretation

- **Text works well alone but adds little to a model that already knows location.** Fine-tuning beats the frozen model at reading rent level from text (out-of-fold Spearman 0.74 vs 0.65), and both beat guessing the mean by a wide margin.
- **Why CV and test disagree:** spatial CV holds out whole neighborhoods, so text (which names the complex and describes its finish level) stands in for missing location knowledge. CV MAE improves $233 → $207. On the test set, 88% of listings sit on a map point the model already knows, and the nearby-rent feature carries that information. Split by location:

| test listings | n | no text | + frozen | + fine-tuned |
|:--|--:|--:|--:|--:|
| at a map point seen in training | 668 | $155 | $158 | $158 |
| at a new map point | 90 | $156 | $159 | $149 |

  The fine-tuned score helps only where location knowledge is missing (about -4%, but n = 90, so this is not conclusive).
- **Text quality drops on the test set** (Spearman 0.47 vs 0.74). Late-2019 listings are more suburban and more alike, so there is less spread for text to explain.
- **Decision:** the Phase 4 model (no text) stays the registered `rent-model`. The text feature brings no reliable gain on held-out listings, and it would add PyTorch plus a 90 MB language model to serving. Note: CV alone would have picked the text model. The override rests on the clean test measurement plus the known CV leak, and it is disclosed here. The text score remains a candidate for properties in neighborhoods with no comps.
