# Machine Learning Labs

Python lab exercises covering matrix operations, plotting, regression, classification, gradient descent, clustering, and neural-network forward propagation.

## Setup

Use a Python virtual environment. The scripts import NumPy, pandas, Matplotlib, seaborn, scikit-learn, imbalanced-learn, and Plotly.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python ForwardPropagation.py
```

Quote lab filenames because they contain spaces and punctuation.

## Data and limitations

Several labs expect local CSV files that are not included: `kc_house_data.csv`, `pima-indians-diabetes.csv`, `games1.csv`, and `loan_data.csv`. The random-forest lab downloads a public dataset over the network. Plotting scripts may need a graphical environment. Syntax and small synthetic examples can be checked independently of external datasets; these checks do not validate model quality.

## License

The repository includes a Apache-2.0 license. See the checked-in license file and retain existing attribution.

## Published site

[Published learning notes](https://arshad9999.github.io/Machine-Learning/). The Pages deployment succeeded and the homepage returned HTTP 200 on 27 October 2026. This does not establish a working capture service or validate model results.
