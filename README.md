# Iris Species Classifier

A small supervised model that guesses an iris species from four measurements. It is trained with k-nearest neighbors and scored only on flowers it did not see.

## How to run

```powershell
python -m pip install -r requirements.txt
python classify.py
```

## Result

On 30 hidden flowers, with `random_state=42`:

- Accuracy: 0.933 (28 of 30)
- Setosa: 10 correct
- Versicolor: 10 correct
- Virginica: 8 correct, 2 called versicolor
- Macro F1: 0.933

## How it works

- The 150 rows come from `sklearn.datasets.load_iris`. There is no CSV.
- 120 rows train the model. 30 are held out.
- `StandardScaler` is fit on the training rows only, then applied to the test rows.
- Each guess is a vote of the 5 nearest training flowers.
- The confusion matrix shows which species were mixed up. Accuracy alone does not.

## Files

| File | What it is |
| --- | --- |
| classify.py | Load, split, scale, train, score |
| requirements.txt | pandas and scikit-learn |
| README.md | This page |
