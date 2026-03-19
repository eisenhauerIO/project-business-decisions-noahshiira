import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def set_random_seed(random_state: int):
    """
    Set random seed for reproducibility.
    """
    np.random.seed(random_state)

def generate_features(n_samples, corr_x4_x5=0.3, p_binary=0.2):
    """Generate feature matrix."""

    X1 = np.random.rand(n_samples)
    X2 = np.random.rand(n_samples)
    X3 = np.random.rand(n_samples)
    X4 = np.random.rand(n_samples)

    noise = np.random.rand(n_samples)
    X5 = corr_x4_x5 * X4 + np.sqrt(1 - corr_x4_x5**2) * noise

    X6 = np.random.choice([0, 1], size=n_samples, p=[1-p_binary, p_binary])
    X7 = np.random.choice([0, 1], size=n_samples, p=[1-p_binary, p_binary])

    return X1, X2, X3, X4, X5, X6, X7


def generate_treatment(n_samples, p_treatment=0.5):
    """Generate treatment assignment."""
    return np.random.choice([0, 1], size=n_samples, p=[1-p_treatment, p_treatment])


def generate_outcome(X1, X2, X3, X4, X5, X6, T, base_prob=0.30):
    """Generate outcome variable with treatment heterogeneity."""

    n_samples = len(T)
    Y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):

        if T[i] == 1:
            prob_Y = (
                base_prob
                - 0.04 * (X1[i] + X2[i])
                + 0.16 * (X3[i] * X4[i])
                - 0.02 * X5[i]
                + 0.02 * X6[i]
            )
        else:
            prob_Y = base_prob

        prob_Y = np.clip(prob_Y, 0, 1)

        Y[i] = np.random.choice([0, 1], p=[1 - prob_Y, prob_Y])

    return Y


def build_dataframe(X1, X2, X3, X4, X5, X6, X7, T, Y):
    """Create pandas dataframe."""

    return pd.DataFrame({
        "X1": X1,
        "X2": X2,
        "X3": X3,
        "X4": X4,
        "X5": X5,
        "X6": X6,
        "X7": X7,
        "T": T,
        "Y": Y
    })


def split_and_save(df, test_size, random_state, output_dir="data"):
    """Split dataset and save."""

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    train_df.to_csv(f"{output_dir}/train_data.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_data.csv", index=False)

    print(f"Data saved to {output_dir}/train_data.csv and {output_dir}/test_data.csv")


def simulate_dataset(
    n_samples=50_000,
    test_size=0.2,
    random_state=2024,
    corr_x4_x5=0.3,
    p_binary=0.2,
    p_treatment=0.5,
    base_prob=0.30,
    output_dir="data"
):
    """
    Umbrella function orchestrating the full simulation pipeline.
    """

    np.random.seed(random_state)

    X1, X2, X3, X4, X5, X6, X7 = generate_features(
        n_samples,
        corr_x4_x5=corr_x4_x5,
        p_binary=p_binary
    )

    T = generate_treatment(
        n_samples,
        p_treatment=p_treatment
    )

    Y = generate_outcome(
        X1, X2, X3, X4, X5, X6,
        T,
        base_prob=base_prob
    )

    df = build_dataframe(X1, X2, X3, X4, X5, X6, X7, T, Y)

    split_and_save(
        df,
        test_size=test_size,
        random_state=random_state,
        output_dir=output_dir
    )


if __name__ == "__main__":
    simulate_dataset()
